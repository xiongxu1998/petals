from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
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
            self._reorder_cache_inplace(cache_tensors, hypo_ids)

            # We chunk the inputs so that peak memory for long sequences fits into `autograd_memory`
            # reserved in `Server._choose_num_blocks()`. This saves us from OOMs if `max_chunk_size_bytes`
            # is at least 4-6x less than `autograd_memory`.
            max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info)
            output_hidden_states = torch.empty_like(hidden_states) if seq_len > max_chunk_length else None
            
            # kv_cache_position_ids = inference_info.kv_cache_position_ids
            # root_position = inference_info.prefix_len
            # if kv_cache_position_ids is None:
            #     root_position = inference_info.prefix_len
            # elif len(kv_cache_position_ids) == 1:
            #     root_position = kv_cache_position_ids[0]
            #     kv_cache_position_ids = None
            # else:
            #     root_position = kv_cache_position_ids[0]
            #     kv_cache_position_ids = kv_cache_position_ids[1: ]
            
            logger.info(f"inference_step, prefix_len: {inference_info.prefix_length}, kv_cache_position_ids: {inference_info.kv_cache_position_ids}")
            layer_past = self._select_layer_past(cache_tensors, inference_info.prefix_length, inference_info.kv_cache_position_ids)
            past_key_values_length = 0
            if layer_past is not None:
                past_key_values_length = layer_past[0].shape[2]
            full_mask = self._create_attention_mask(
                tree_attention_mask=inference_info.tree_attention_mask,
                src_len=seq_len + past_key_values_length,
                past_key_values_length=past_key_values_length,
                device=hidden_states.device,
            )
            attention_mask = self.convert_mask_to_scores(full_mask)
            # logger.info(f"inference_step, full_mask: {attention_mask}")
            logger.info(f"inference_step, layer_past: {layer_past}")
            for offset in range(0, seq_len, max_chunk_length):
                hidden_states_chunk = hidden_states[:, offset : offset + max_chunk_length, :]
                
                # if full_mask is not None:
                #     attention_mask = full_mask[:, :inference_info.prefix_length + offset + chunk_len]
                # else:
                #     attention_mask = None
                output_hidden_states_chunk, new_kvs = self.module.forward(
                    hidden_states_chunk, layer_past=layer_past, attention_mask=attention_mask, use_cache=True
                )
                if seq_len > max_chunk_length:
                    output_hidden_states[:, offset : offset + max_chunk_length] = output_hidden_states_chunk
                else:
                    output_hidden_states = output_hidden_states_chunk  # saves one memcopy
                layer_past = new_kvs

            self._update_cache_inplace(cache_tensors, new_kvs, past_key_values_length)
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
                
    def _create_attention_mask(
        self,
        tree_attention_mask: Optional[torch.Tensor],
        *,
        src_len: int,                # prefix_len + tree_len
        past_key_values_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if tree_attention_mask is None or is_dummy(tree_attention_mask):
            return None

        # ---- 1. 解包树段 ----
        if tree_attention_mask.dtype != torch.uint8:
            raise TypeError("tree_attention_mask should be uint8 packed")

        if hasattr(torch, "unpackbits"):
            bits = torch.unpackbits(tree_attention_mask.to(device), dim=-1)
        else:
            bits = self._unpackbits_fallback(tree_attention_mask.to(device), dim=-1)

        # bits: [B, tree_len, n_chunks, 64]
        # logger.info(f"bits: {bits}")
        bits = bits.flatten(start_dim=-3)            # [B, tree_len, n_chunks*64]
        tree_len = bits.size(1)
        tree_mask = bits[..., :tree_len].bool()      # [B, tree_len, tree_len]

        # ---- 2. 计算前缀长度并构造完整mask ----
        prefix_len = src_len - tree_len
        B = tree_mask.size(0)

        # 构造完整的attention mask [B, src_len, src_len]
        full_mask = torch.zeros(B, src_len, src_len, dtype=torch.bool, device=device)
        
        # 前缀部分：标准下三角矩阵（严格因果性）
        # 第0行：只有[0]位置为True（只能看到自己）
        # 第1行：只有[0,1]位置为True（能看到前面的和自己）
        # 以此类推...
        if prefix_len > 0:
            prefix_mask = torch.tril(torch.ones(prefix_len, prefix_len, dtype=torch.bool, device=device))
            full_mask[:, :prefix_len, :prefix_len] = prefix_mask
            
            # 前缀对树段的可见性：前缀token不能看到后面的树段token（保持因果性）
            # full_mask[:, :prefix_len, prefix_len:] 保持为False（已经初始化为0）
        
        # 树段对前缀的可见性：树段的每个token都能看到所有前缀token
        if prefix_len > 0 and tree_len > 0:
            full_mask[:, prefix_len:, :prefix_len] = True
        
        # 树段内部的可见性：使用解包后的tree_mask
        if tree_len > 0:
            full_mask[:, prefix_len:, prefix_len:] = tree_mask

        # ---- 3. 只保留当前tokens对应的行 ----
        # 当前token数量 = src_len - past_key_values_length
        current_token_count = src_len - past_key_values_length
        
        if current_token_count <= 0:
            # 如果没有当前tokens，返回None
            return None
        
        # 只保留最后current_token_count行
        # 这些行对应当前step需要计算attention的tokens
        current_mask = full_mask[:, -current_token_count:, :]  # [B, current_token_count, src_len]
        
        logger.info(f"Original mask shape: {full_mask.shape}, "
                    f"Current mask shape: {current_mask.shape}, "
                    f"src_len: {src_len}, past_key_values_length: {past_key_values_length}")

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
        
    def _unpackbits_fallback(self, x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        """
        手动实现 torch.unpackbits, 保持与 numpy 默认的 'big' bitorder 一致：
        MSB 在前 → 对应 shift 7,6,...,0
        支持 GPU & broadcast, 仅限 uint8。
        """
        if x.dtype != torch.uint8:
            raise TypeError("fallback unpackbits expects uint8 input")

        # 把目标维度挪到最后，方便向量化
        if dim != -1 and dim != x.ndim - 1:
            x = x.movedim(dim, -1)

        shifts = torch.arange(7, -1, -1, device=x.device, dtype=torch.uint8)
        bits = (x.unsqueeze(-1) >> shifts) & 1          # [..., 8]
        # 现在 bits 的最后一维是 bit 列表；如果原 dim 不是最后，再挪回去
        if dim != -1 and dim != x.ndim - 1:
            bits = bits.movedim(-2, dim)

        return bits

    def _select_layer_past(self, cache_tensors: Sequence[torch.Tensor], prefix_length: int, kv_cache_position_ids: Optional[torch.Tensor] = None) -> Sequence[torch.Tensor]:
        """Extract first {prefix_length} tokens and optionally specific positions based on kv_cache_position_ids"""
        key_cache, value_cache = list(cache_tensors[0::2]), list(cache_tensors[1::2])
        
        for i in range(len(key_cache)):
            # 首先获取原始的 key 和 value cache
            key_cache[i] = key_cache[i].flatten(0, 1)  # [batch * num_kv_heads, head_dim, total_length]
            value_cache[i] = value_cache[i].flatten(0, 1)  # [batch * num_kv_heads, total_length, head_dim]
            
            k, v = key_cache[i], value_cache[i]
            
            # 如果提供了 kv_cache_position_ids，需要选择特定位置的 cache
            if kv_cache_position_ids is not None and not is_dummy(kv_cache_position_ids):
                logger.info(f"Selecting KV cache using position_ids: {kv_cache_position_ids}")
                
                # kv_cache_position_ids 的形状应该是 [batch_size, num_positions] 或 [num_positions]
                position_ids = kv_cache_position_ids.to(k.device)
                
                # 确保 position_ids 是 2D 的
                if position_ids.dim() == 1:
                    # 如果是 1D，假设 batch_size = 1
                    position_ids = position_ids.unsqueeze(0)  # [1, num_positions]
                
                batch_size = position_ids.shape[0]
                num_tree_positions = position_ids.shape[1]
                num_kv_heads = k.shape[0] // batch_size
                
                # 构建完整的位置列表：前文 + 树节点
                all_positions_list = []
                for batch_idx in range(batch_size):
                    batch_positions = position_ids[batch_idx]  # [num_tree_positions]
                    
                    # 根节点位置是第0个元素
                    root_position = batch_positions[0].item()
                    
                    # 前文位置：0 到 root_position-1
                    prefix_positions = torch.arange(0, root_position, device=position_ids.device)
                    
                    # 合并前文位置和树位置
                    complete_positions = torch.cat([prefix_positions, batch_positions])
                    logger.info(f"_select_layer_past, complete_positions: {complete_positions}")
                    all_positions_list.append(complete_positions)
                
                # 找到最大长度，用于填充
                max_length = max(len(pos) for pos in all_positions_list)
                
                # 创建填充后的位置张量
                padded_positions = torch.zeros(batch_size, max_length, dtype=torch.long, device=position_ids.device)
                position_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=position_ids.device)
                
                for batch_idx, positions in enumerate(all_positions_list):
                    seq_len = len(positions)
                    padded_positions[batch_idx, :seq_len] = positions
                    position_mask[batch_idx, :seq_len] = True
                
                # 确保位置索引在有效范围内
                padded_positions = torch.clamp(padded_positions, 0, k.shape[2] - 1)
                
                # 展开 position_ids 以匹配所有头
                expanded_positions = padded_positions.repeat_interleave(num_kv_heads, dim=0)  # [batch*num_kv_heads, max_length]
                expanded_mask = position_mask.repeat_interleave(num_kv_heads, dim=0)  # [batch*num_kv_heads, max_length]
                
                # 使用 gather 操作选择对应位置的 cache
                # key cache: 在第2维(seq_len维)上选择
                selected_key = torch.gather(k, 2, expanded_positions.unsqueeze(1).expand(-1, k.shape[1], -1))
                
                # value cache: 在第1维(seq_len维)上选择  
                selected_value = torch.gather(v, 1, expanded_positions.unsqueeze(2).expand(-1, -1, v.shape[2]))
                
                # 应用mask，将无效位置设为0（虽然通常不会被使用）
                if expanded_mask.any():
                    mask_key = expanded_mask.unsqueeze(1).expand(-1, k.shape[1], -1)
                    mask_value = expanded_mask.unsqueeze(2).expand(-1, -1, v.shape[2])
                    
                    # 使用原始tensor的数据类型，而不是强制转换为float
                    selected_key = selected_key * mask_key.to(selected_key.dtype)
                    selected_value = selected_value * mask_value.to(selected_value.dtype)
                
                key_cache[i] = selected_key
                value_cache[i] = selected_value
                
                logger.info(
                    f"[KV] L{i:02d} selected key shape {tuple(selected_key.shape)} "
                    f"value shape {tuple(selected_value.shape)} "
                    f"root_positions: {[pos[0].item() for pos in all_positions_list]} "
                    f"total_positions: {[len(pos) for pos in all_positions_list]}"
                )
            else:
                # 原有逻辑：只选择前 prefix_length 个 tokens
                key_cache[i] = k[:, :, :prefix_length]
                value_cache[i] = v[:, :prefix_length, :]
                
                logger.info(
                    f"[KV] L{i:02d} prefix key shape {tuple(key_cache[i].shape)} "
                    f"value shape {tuple(value_cache[i].shape)} "
                    f"prefix_length={prefix_length}"
                )
            
            # 打印调试信息
            k, v = key_cache[i], value_cache[i]
            sample_k = k.flatten()[0].item() if k.numel() else float('nan')
            sample_v = v.flatten()[0].item() if v.numel() else float('nan')
            logger.info(
                f"[KV] L{i:02d} final key shape {tuple(k.shape)} "
                f"value shape {tuple(v.shape)} "
                f"sample k={sample_k:.4g} v={sample_v:.4g}"
            )
        
        layer_past = tuple(chain(*zip(key_cache, value_cache)))
        logger.info(f"cache_tensors size: {len(cache_tensors)}, selected layer_past size: {len(layer_past)}")
        
        return PerDeviceTensors(*layer_past) if len(self.module.module_shards) > 1 else layer_past

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
