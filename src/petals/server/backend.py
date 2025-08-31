from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import time
import numpy as np
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from tensor_parallel import TensorParallel
from tensor_parallel.tensor_parallel import PerDeviceTensors
from transformers import PretrainedConfig

from petals.data_structures import InferenceMetadata
from petals.server.memory_cache import MemoryCache
from petals.server.task_pool import PrioritizedTaskPool
from petals.utils.misc import get_size_in_bytes, is_dummy

logger = get_logger(__name__)


class TransformerBackend(ModuleBackend):
    """A wrapper for a transformer block that can process requests for forward, backward and inference"""

    _peft_module = None

    def __init__(
        self,
        *args,
        config: PretrainedConfig,
        memory_cache: MemoryCache,
        backend_dtype: torch.dtype,
        max_chunk_size_bytes: int,
        **kwargs,
    ):
        import petals.utils.peft as _peft_module

        self._peft_module = _peft_module

        super().__init__(*args, **kwargs)
        assert isinstance(self.module, TensorParallel)
        self.config = config
        self.memory_cache = memory_cache
        self.max_chunk_size_bytes = max_chunk_size_bytes

        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"

        max_batch_size = self.forward_pool.max_batch_size
        device = self.module.devices[self.module.output_device_index]
        self.inference_pool = PrioritizedTaskPool(
            self.inference_step, max_batch_size=max_batch_size, device=device, name=f"{self.name}_inference"
        )  # note: inference_pools may be merged later, see merge_inference_pools_inplace
        self.forward_pool = PrioritizedTaskPool(
            self.forward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_forward"
        )
        self.backward_pool = PrioritizedTaskPool(
            self.backward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_backward"
        )

        self.dtype = backend_dtype
        self.dtype_bytes = get_size_in_bytes(self.dtype)
        self.shard_num_heads = []
        for shard in self.module.module_shards:
            for submodule in shard.modules():
                if isinstance(submodule, config.attn_class):
                    self.shard_num_heads.append(submodule.num_heads)
        assert len(self.shard_num_heads) == len(self.module.devices)
        assert sum(self.shard_num_heads) == config.num_attention_heads

        self.inference_schema = (
            (
                *self.args_schema,
                BatchTensorDescriptor((), dtype=self.dtype),
                BatchTensorDescriptor((), dtype=torch.int64),
            ),
            self.kwargs_schema,
        )

        self.cache_bytes_per_token: Dict[torch.device, int] = Counter()
        for descr in self.get_inference_cache_descriptors(batch_size=1, max_length=1):
            self.cache_bytes_per_token[descr.device] += descr.numel() * get_size_in_bytes(descr.dtype)
        
        # 缓存解析后的 tree_attention_mask
        self._tree_mask_cache: Dict[int, torch.Tensor] = {}  # key: hash of packed mask, value: unpacked tree mask

    def get_inference_cache_descriptors(self, batch_size: int, max_length: int) -> Sequence[TensorDescriptor]:
        """Create tensor descriptors for attention cache tensors used during inference_step"""
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        cache_tensors = []
        for device, num_heads in zip(self.module.devices, self.shard_num_heads):
            num_heads //= self.config.num_key_value_groups
            if hasattr(self.config, "num_key_value_heads"):
                num_heads = self.config.num_key_value_heads
            keys = TensorDescriptor((batch_size, num_heads, head_dim, max_length), dtype=self.dtype, device=device)
            values = TensorDescriptor((batch_size, num_heads, max_length, head_dim), dtype=self.dtype, device=device)
            cache_tensors.extend((keys, values))
        return cache_tensors

    def forward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().forward(*inputs)

    def backward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().backward(*inputs)

    @torch.inference_mode()
    def inference_step(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_info: InferenceMetadata,
    ) -> Tuple[torch.Tensor, ...]:
        assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"
        seq_len = hidden_states.shape[1]

        with self.memory_cache.use_cache(
            *inference_info.cache_handles
        ) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter):
            # 时间记录
            # t0 = time.time()
            
            self._reorder_cache_inplace(cache_tensors, hypo_ids)
            # t1 = time.time()
            
            # 估算最大块长度
            max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info)
            output_hidden_states = torch.empty_like(hidden_states) if seq_len > max_chunk_length else None
            # t2 = time.time()
            
            # logger.info(f"inference_step, prefix_len: {inference_info.prefix_length}, kv_cache_position_ids: {inference_info.kv_cache_position_ids}")
            
            # 1. 选择需要的 cache（返回两个值）
            layer_past, need_reorder = self._select_layer_past(
                cache_tensors, 
                inference_info.prefix_length, 
                inference_info.kv_cache_position_ids
            )
            # t3 = time.time()
            
            # 2. 获取 past_key_values 的长度
            past_key_values_length = 0
            if layer_past is not None and len(layer_past) > 0:
                past_key_values_length = layer_past[0].shape[2]
            
            # 3. 如果使用了 kv_cache_position_ids，需要更新 cache_tensors
            # 使选中的 cache 在原始位置上变成连续存储
            if need_reorder:
                # 这里需要将 layer_past 的内容写回 cache_tensors 的前面部分
                # 使其在物理上连续存储
                self._compact_cache_inplace(cache_tensors, layer_past, past_key_values_length)
            # t4 = time.time()
            
            # 4. 创建 attention mask
            full_mask = self._create_attention_mask(
                tree_attention_mask=inference_info.tree_attention_mask,
                src_len=seq_len + past_key_values_length,
                past_key_values_length=past_key_values_length,
                device=hidden_states.device,
            )
            attention_mask = self.convert_mask_to_scores(full_mask) if full_mask is not None else None
            # t5 = time.time()
            logger.info(f"full_mask: {full_mask}")
            # logger.info(f"inference_step, layer_past: {layer_past}")
            
            # 5. 分块处理
            forward_time = 0
            for offset in range(0, seq_len, max_chunk_length):
                hidden_states_chunk = hidden_states[:, offset : offset + max_chunk_length, :]
                
                # t_forward_start = time.time()
                output_hidden_states_chunk, new_kvs = self.module.forward(
                    hidden_states_chunk, 
                    layer_past=layer_past, 
                    attention_mask=attention_mask, 
                    use_cache=True
                )
                # forward_time += time.time() - t_forward_start
                
                if seq_len > max_chunk_length:
                    output_hidden_states[:, offset : offset + max_chunk_length] = output_hidden_states_chunk
                else:
                    output_hidden_states = output_hidden_states_chunk
                
                layer_past = new_kvs
            # t6 = time.time()

            # 6. 最后更新 cache（将新生成的 KV 写入）
            self._update_cache_inplace(cache_tensors, new_kvs, past_key_values_length)
            # t7 = time.time()
            
            # 打印时间统计
            # logger.info(
            #     f"[Timing] inference_step total: {t7-t0:.4f}s | "
            #     f"reorder_cache: {t1-t0:.4f}s | "
            #     f"estimate_chunk: {t2-t1:.4f}s | "
            #     f"select_layer_past: {t3-t2:.4f}s | "
            #     f"compact_cache: {t4-t3:.4f}s | "
            #     f"create_mask: {t5-t4:.4f}s | "
            #     f"forward: {forward_time:.4f}s | "
            #     f"update_cache: {t7-t6:.4f}s"
            # )
            
            return (output_hidden_states,)

    def _estimate_max_chunk_length(self, hidden_states: torch.Tensor, inference_info: InferenceMetadata) -> int:
        # We assume that attention logit matrices are the main thing that consumes memory, given that
        # the model uses multi-query attention
        batch_size, seq_length, hidden_size = hidden_states.shape
        worst_case_length = inference_info.prefix_length + seq_length
        attn_bytes_per_token = max(self.shard_num_heads) * batch_size * self.dtype_bytes * worst_case_length
        return max(1, self.max_chunk_size_bytes // attn_bytes_per_token)

    def _reorder_cache_inplace(self, cache_tensors: torch.Tensor, hypo_ids: torch.Tensor):
        """If hypo_ids is specified, reorder elements of each cache tensor in-place by taking indices from hypo_ids"""
        if not is_dummy(hypo_ids):
            for cache_tensor in cache_tensors:
                cache_tensor[...] = cache_tensor[hypo_ids.to(cache_tensor.device)]  # in-place reorder cache by hypo ids
    
    def _get_tree_mask_from_cache(self, tree_attention_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        """从缓存中获取解析后的 tree mask，如果不存在则解析并缓存"""
        # 计算 tree_attention_mask 的哈希值作为缓存键
        # 使用 tensor 的数据指针和形状作为键，因为内容相同的 tensor 会有相同的表示
        cache_key = hash((tree_attention_mask.data_ptr(), tree_attention_mask.shape, tree_attention_mask.stride()))
        
        if cache_key in self._tree_mask_cache:
            # 从缓存中获取并移动到正确的设备
            cached_mask = self._tree_mask_cache[cache_key]
            if cached_mask.device != device:
                cached_mask = cached_mask.to(device)
            # logger.info(f"Using cached tree mask for key {cache_key}")
            return cached_mask
        
        tree_mask = self._unpackbits_fallback(tree_attention_mask.to(device))
        
        # 缓存解析后的 mask
        self._tree_mask_cache[cache_key] = tree_mask  # 存储在 CPU 上以节省 GPU 内存
        # logger.info(f"Cached new tree mask for key {cache_key}, cache size: {len(self._tree_mask_cache)}")
        
        return tree_mask


    def _create_attention_mask(
        self,
        tree_attention_mask: Optional[torch.Tensor],
        *,
        src_len: int,                # prefix_len + tree_len
        past_key_values_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        # start_time = time.time()
        
        if tree_attention_mask is None or is_dummy(tree_attention_mask):
            return None

        # ---- 1. 解包树段（使用缓存）----
        # unpack_start = time.time()
        if tree_attention_mask.dtype != torch.uint8:
            raise TypeError("tree_attention_mask should be uint8 packed")

        # 从缓存中获取解析后的 tree_mask
        tree_mask = self._get_tree_mask_from_cache(tree_attention_mask, device)
        tree_len = tree_mask.size(1)
        B = tree_mask.size(0)
        prefix_len = src_len - tree_len
        current_token_count = src_len - past_key_values_length
        
        logger.info(f"current_token_count: {current_token_count}, past_key_values_length: {past_key_values_length}, prefix_len: {prefix_len}")
        if current_token_count <= 0:
            return None
        if past_key_values_length == 0:
            full_mask = torch.zeros(B, src_len, src_len, dtype=torch.bool, device=device)
            if prefix_len > 0:
                causal_indices = torch.tril_indices(prefix_len, prefix_len, device=device)
                full_mask[:, causal_indices[0], causal_indices[1]] = True
            if prefix_len > 0 and tree_len > 0:
                full_mask[:, prefix_len:, :prefix_len] = True
            if tree_len > 0:
                full_mask[:, prefix_len:, prefix_len:] = tree_mask
            return full_mask
        
        # 情况B: 后续的增量生成 (通常 current_token_count 很小)
        else:
            current_mask = torch.zeros(B, current_token_count, src_len, dtype=torch.bool, device=device)
            start_pos = past_key_values_length
            if start_pos < prefix_len:
                prefix_tokens = min(current_token_count, prefix_len - start_pos)
                # 因果mask
                for i in range(prefix_tokens):
                    current_mask[:, i, :start_pos + i + 1] = True
                if current_token_count > prefix_tokens:
                    tree_tokens = current_token_count - prefix_tokens
                    current_mask[:, prefix_tokens:, :prefix_len] = True
                    current_mask[:, prefix_tokens:, prefix_len:] = tree_mask[:, :tree_tokens, :]
            else:
                tree_start = start_pos - prefix_len
                if prefix_len > 0:
                    current_mask[:, :, :prefix_len] = True
                current_mask[:, :, prefix_len:] = tree_mask[:, tree_start:tree_start + current_token_count, :]
            return current_mask
    
    def convert_mask_to_scores(self, mask: torch.Tensor) -> torch.Tensor:
        """
        将布尔attention mask转换为attention分数
        
        Args:
            mask: 布尔tensor，True表示可见，False表示不可见
            
        Returns:
            转换后的tensor，True->0.0, False->-65504.0
        """
        if mask.dtype != torch.bool:
            raise TypeError(f"Expected bool tensor, got {mask.dtype}")
        
        # 创建与输入相同形状的浮点tensor
        scores = torch.full_like(mask, -65504.0, dtype=torch.float)
        
        # True的位置设为0.0
        scores[mask] = 0.0
        
        return scores
        
    def _unpackbits_fallback(self, packed: torch.Tensor) -> torch.Tensor:
        batch_size = packed.shape[0]
        n = packed.shape[1]  # 从第二维获取原始矩阵大小
        
        # 只取第一个"行"的数据（因为打包时复制了）
        # (batch_size, n, num_int64, 8) -> (batch_size, num_int64, 8)
        packed_data = packed[:, 0, :, :]
        
        # 展平为字节序列
        # (batch_size, num_int64, 8) -> (batch_size, num_bytes)
        batch_size, n, num_int64, _ = packed.shape
    
        # 重塑为按行的字节序列
        # (batch_size, n, num_int64, 8) -> (batch_size, n, num_int64*8)
        packed_bytes = packed.reshape(batch_size, n, num_int64 * 8)
        
        # 转换为numpy
        packed_np = packed_bytes.cpu().numpy().astype(np.uint8)
        
        # 按行解包 - unpackbits会将每行的字节解包为bits
        unpacked = np.unpackbits(packed_np, axis=-1)
        
        # unpacked现在的shape是 (batch_size, n, num_int64*8*8)
        # 只保留前n个bits（每行）
        unpacked = unpacked[:, :, :n]
        
        # shape已经是 (batch_size, n, n)
        mask_bool = torch.from_numpy(unpacked.astype(bool)).to(packed.device)
        
        return mask_bool
    
    def _compact_cache_inplace(
        self, 
        cache_tensors: Sequence[torch.Tensor], 
        selected_cache: Sequence[torch.Tensor], 
        selected_length: int
    ):
        """
        将选中的 cache 内容写回原始 cache_tensors 的前面部分，
        使其在物理上连续存储
        """
        for i, (cache_tensor, selected) in enumerate(zip(cache_tensors, selected_cache)):
            if i % 2 == 0:  # Key cache
                # selected shape: [batch * num_heads, head_dim, selected_length]
                # cache_tensor shape: [batch, num_heads, head_dim, max_length]
                batch_size = cache_tensor.shape[0]
                num_heads = cache_tensor.shape[1]
                head_dim = cache_tensor.shape[2]
                
                selected_reshaped = selected.view(batch_size, num_heads, head_dim, selected_length)
                cache_tensor[:, :, :, :selected_length] = selected_reshaped
            else:  # Value cache
                # selected shape: [batch * num_heads, selected_length, head_dim]
                # cache_tensor shape: [batch, num_heads, max_length, head_dim]
                batch_size = cache_tensor.shape[0]
                num_heads = cache_tensor.shape[1]
                head_dim = cache_tensor.shape[3]
                
                selected_reshaped = selected.view(batch_size, num_heads, selected_length, head_dim)
                cache_tensor[:, :, :selected_length, :] = selected_reshaped

    def _select_layer_past(self, cache_tensors: Sequence[torch.Tensor], prefix_length: int, kv_cache_position_ids: Optional[torch.Tensor] = None) -> Tuple[Sequence[torch.Tensor], bool]:
        """Extract first {prefix_length} tokens and optionally specific positions based on kv_cache_position_ids"""
        # start_time = time.time()
        key_cache, value_cache = list(cache_tensors[0::2]), list(cache_tensors[1::2])
        need_reorder = False
        
        # 快速路径：如果没有position_ids，直接切片
        if kv_cache_position_ids is None or is_dummy(kv_cache_position_ids):
            for i in range(len(key_cache)):
                k = key_cache[i].flatten(0, 1)
                v = value_cache[i].flatten(0, 1)
                
                key_cache[i] = k[:, :, :prefix_length]
                value_cache[i] = v[:, :prefix_length, :]
        else:
            # 预处理 position_ids
            position_ids = kv_cache_position_ids
            # if position_ids.dim() == 1:
            #     position_ids = position_ids.unsqueeze(0)
            
            batch_size = 1
            
            # 提取第一个batch的tree positions
            first_batch = position_ids
            if first_batch.numel() == 1:
                # 如果只有一个元素，直接取值
                tree_positions = [first_batch.item()]
            else:
                # 如果有多个元素，转换为列表
                tree_positions = first_batch.cpu().tolist()
            
            root_position = tree_positions[0]  # 第一个位置是root
            
            # 构建完整位置：前缀[0, 1, ..., root-1] + tree positions
            prefix_positions = list(range(root_position))  # [0, 1, 2, ..., root-1]
            complete_positions = prefix_positions + tree_positions  # 完整序列
            
            # 检查完整序列是否连续（从0开始）
            expected_continuous = list(range(len(complete_positions)))
            is_continuous = complete_positions == expected_continuous
            
            if is_continuous:
                # 连续情况：直接切片，类似prefix_length的处理方式
                seq_length = len(complete_positions)
                for i in range(len(key_cache)):
                    k = key_cache[i].flatten(0, 1)
                    v = value_cache[i].flatten(0, 1)
                    
                    key_cache[i] = k[:, :, :seq_length]
                    value_cache[i] = v[:, :seq_length, :]
            else:
                # 非连续情况：使用index_select
                need_reorder = True
                
                # 检查是否所有batch的位置相同
                all_same = batch_size == 1
                
                if all_same:
                    # 所有batch位置相同，可以共享索引
                    positions_tensor = torch.tensor(complete_positions, device=cache_tensors[0].device)
                    
                    for i in range(len(key_cache)):
                        k = key_cache[i].flatten(0, 1)
                        v = value_cache[i].flatten(0, 1)
                        
                        # 使用index_select
                        key_cache[i] = k.index_select(2, positions_tensor)
                        value_cache[i] = v.index_select(1, positions_tensor)
                else:
                    # 不同batch有不同位置，需要更复杂的处理
                    for i in range(len(key_cache)):
                        k = key_cache[i].flatten(0, 1)
                        v = value_cache[i].flatten(0, 1)
                        num_kv_heads = k.shape[0] // batch_size
                        
                        # 为每个batch单独处理
                        selected_keys = []
                        selected_values = []
                        max_length = 0
                        
                        for batch_idx in range(batch_size):
                            batch_tensor = position_ids[batch_idx]
                            if batch_tensor.numel() == 1:
                                batch_tree_positions = [batch_tensor.item()]
                            else:
                                batch_tree_positions = batch_tensor.cpu().tolist()
                            
                            batch_root = batch_tree_positions[0]
                            batch_prefix = list(range(batch_root))
                            batch_complete = batch_prefix + batch_tree_positions
                            batch_positions_tensor = torch.tensor(batch_complete, device=position_ids.device)
                            max_length = max(max_length, len(batch_complete))
                            
                            for head_idx in range(num_kv_heads):
                                idx = batch_idx * num_kv_heads + head_idx
                                selected_keys.append(k[idx:idx+1].index_select(2, batch_positions_tensor))
                                selected_values.append(v[idx:idx+1].index_select(1, batch_positions_tensor))
                        
                        # 合并结果
                        if all(sk.shape[2] == selected_keys[0].shape[2] for sk in selected_keys):
                            # 如果所有序列长度相同，可以直接cat
                            key_cache[i] = torch.cat(selected_keys, dim=0)
                            value_cache[i] = torch.cat(selected_values, dim=0)
                        else:
                            # 需要padding到相同长度
                            padded_key = torch.zeros(k.shape[0], k.shape[1], max_length, dtype=k.dtype, device=k.device)
                            padded_value = torch.zeros(v.shape[0], max_length, v.shape[2], dtype=v.dtype, device=v.device)
                            
                            for idx, (sk, sv) in enumerate(zip(selected_keys, selected_values)):
                                seq_len = sk.shape[2]
                                padded_key[idx, :, :seq_len] = sk[0]
                                padded_value[idx, :seq_len, :] = sv[0]
                            
                            key_cache[i] = padded_key
                            value_cache[i] = padded_value
        
        layer_past = tuple(chain(*zip(key_cache, value_cache)))
        
        # 返回 cache 和新的长度信息
        result = PerDeviceTensors(*layer_past) if len(self.module.module_shards) > 1 else layer_past
        
        return result, need_reorder

    def _update_cache_inplace(
        self, cache_tensors: Sequence[torch.Tensor], new_kvs: Sequence[torch.Tensor], prefix_length: int
    ):
        """Writes new key/value tensors back into cache, works in-place"""
        _batch_size_times_num_kv_heads, head_dim, new_length = new_kvs[0].shape
        for cache_key, new_key in zip(cache_tensors[0::2], new_kvs[0::2]):
            new_key = new_key.view(*cache_key.shape[:3], new_length)
            cache_key[:, :, :, prefix_length:new_length] = new_key[:, :, :, prefix_length:new_length]
        for cache_value, new_value in zip(cache_tensors[1::2], new_kvs[1::2]):
            new_value = new_value.view(*cache_value.shape[:2], new_length, head_dim)
            cache_value[:, :, prefix_length:new_length, :] = new_value[:, :, prefix_length:new_length, :]

    def get_pools(self) -> Sequence[PrioritizedTaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool

    def get_info(self) -> Dict[str, Any]:
        """Get module parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)

    def shutdown(self):
        # Break the cyclic references, otherwise TransformerBackend may be not garbage-collected
        self.forward_pool = self.backward_pool = self.inference_pool = None

        # Explicitly free the GPU memory. This is not necessary at the time this code is written,
        # but may help to avoid future issues when the module is not garbage-collected for some reasons
        dummy = torch.tensor([])
        for p in self.module.parameters():
            p.data = dummy


def merge_inference_pools_inplace(backends: Dict[ExpertUID, TransformerBackend]):
    """Replace each backend's rpc_inference pools with a combined pool runs multiple blocks in one call"""
    assert len(backends) != 0 and all(isinstance(b, TransformerBackend) for b in backends.values())
    first_pool = next(iter(backends.values())).inference_pool
    merged_pool = PrioritizedTaskPool(
        _MergedInferenceStep(backends),
        max_batch_size=first_pool.max_batch_size,
        device=first_pool.device,
        name=f"merged_inference",
    )
    for backend in backends.values():
        assert not backend.inference_pool.is_alive()
        backend.inference_pool = merged_pool


class _MergedInferenceStep:
    def __init__(self, backends: Dict[ExpertUID, TransformerBackend]):
        self.backends = backends

    @torch.inference_mode()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_infos: Sequence[InferenceMetadata],
        *optional_prompts: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        assert len(inference_infos) == len(
            optional_prompts
        ), f"found {len(inference_infos)} blocks but {len(optional_prompts)} prompts"
        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if optional_prompt is not None:
                hidden_states[:, : optional_prompt.shape[1]] += optional_prompt
            (hidden_states,) = self.backends[inference_info.uid].inference_step(hidden_states, hypo_ids, inference_info)
        return (hidden_states,)
