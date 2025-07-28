from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
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
            
        # 性能分析相关
        self._cached_mask = None
        self._cached_mask_src_len = 0
        self._timing_stats: Dict[str, List[float]] = {}
        self._enable_profiling = True

    @contextmanager
    def _profile_section(self, name: str):
        """用于性能分析的上下文管理器"""

            
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            if name not in self._timing_stats:
                self._timing_stats[name] = []
            self._timing_stats[name].append(elapsed)
            
    def print_timing_stats(self):
        """打印性能统计信息"""
        if not self._timing_stats:
            return
            
        logger.info("=== Performance Statistics ===")
        for name, times in sorted(self._timing_stats.items()):
            times_array = np.array(times)
            logger.info(
                f"{name}: "
                f"mean={times_array.mean():.4f}s, "
                f"std={times_array.std():.4f}s, "
                f"min={times_array.min():.4f}s, "
                f"max={times_array.max():.4f}s, "
                f"total={times_array.sum():.4f}s, "
                f"count={len(times)}"
            )

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
        with self._profile_section("inference_step_total"):
            assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"
            seq_len = hidden_states.shape[1]

            with self.memory_cache.use_cache(
                *inference_info.cache_handles
            ) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter):
                
                # 1. Reorder cache
                with self._profile_section("reorder_cache"):
                    self._reorder_cache_inplace(cache_tensors, hypo_ids)

                # 2. 估算最大块长度
                with self._profile_section("estimate_chunk_length"):
                    max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info)
                    output_hidden_states = torch.empty_like(hidden_states) if seq_len > max_chunk_length else None
                
                logger.info(f"inference_step, prefix_len: {inference_info.prefix_length}, kv_cache_position_ids: {inference_info.kv_cache_position_ids}")
                
                # 3. 选择需要的 cache
                with self._profile_section("select_layer_past"):
                    layer_past, new_position_mapping = self._select_layer_past(
                        cache_tensors, 
                        inference_info.prefix_length, 
                        inference_info.kv_cache_position_ids
                    )
                
                # 4. 获取 past_key_values 的长度
                past_key_values_length = 0
                if layer_past is not None and len(layer_past) > 0:
                    past_key_values_length = layer_past[0].shape[2]
                
                # 5. Compact cache if needed
                if inference_info.kv_cache_position_ids is not None and not is_dummy(inference_info.kv_cache_position_ids):
                    with self._profile_section("compact_cache"):
                        self._compact_cache_inplace(cache_tensors, layer_past, past_key_values_length)
                
                # 6. 创建 attention mask
                with self._profile_section("create_attention_mask"):
                    full_mask = self._create_attention_mask(
                        tree_attention_mask=inference_info.tree_attention_mask,
                        src_len=seq_len + past_key_values_length,
                        past_key_values_length=past_key_values_length,
                        device=hidden_states.device,
                    )
                    
                with self._profile_section("convert_mask_to_scores"):
                    attention_mask = self.convert_mask_to_scores(full_mask) if full_mask is not None else None
                
                # 7. 分块处理
                with self._profile_section("forward_chunks"):
                    for offset in range(0, seq_len, max_chunk_length):
                        hidden_states_chunk = hidden_states[:, offset : offset + max_chunk_length, :]
                        
                        with self._profile_section("module_forward"):
                            output_hidden_states_chunk, new_kvs = self.module.forward(
                                hidden_states_chunk, 
                                layer_past=layer_past, 
                                attention_mask=attention_mask, 
                                use_cache=True
                            )
                        
                        if seq_len > max_chunk_length:
                            output_hidden_states[:, offset : offset + max_chunk_length] = output_hidden_states_chunk
                        else:
                            output_hidden_states = output_hidden_states_chunk
                        
                        layer_past = new_kvs

                # 8. 更新 cache
                with self._profile_section("update_cache"):
                    self._update_cache_inplace(cache_tensors, new_kvs, past_key_values_length)
                
                
                self.print_timing_stats()
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
        src_len: int,
        past_key_values_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if tree_attention_mask is None or is_dummy(tree_attention_mask):
            return None

        # 解包tree mask
        if hasattr(torch, "unpackbits"):
            bits = torch.unpackbits(tree_attention_mask.to(device), dim=-1)
        else:
            bits = self._unpackbits_fallback(tree_attention_mask.to(device), dim=-1)
        
        bits = bits.flatten(start_dim=-3)
        tree_len = bits.size(1)
        tree_template = bits[..., :tree_len].bool()
        
        prefix_len = src_len - tree_len
        B = tree_template.size(0)
        
        current_token_count = src_len - past_key_values_length
        if current_token_count <= 0:
            return None
        
        # 检查是否需要更新缓存的mask
        if (self._cached_mask is None or 
            self._cached_mask_src_len < src_len or
            self._cached_mask.shape[0] != B or
            self._cached_mask.device != device):
            
            if (self._cached_mask is not None and 
                self._cached_mask_src_len < src_len and
                self._cached_mask.shape[0] == B and
                self._cached_mask.device == device):
                
                # 基于缓存扩展
                old_len = self._cached_mask_src_len
                new_mask = torch.zeros(B, src_len, src_len, dtype=torch.bool, device=device)
                
                # 复制旧的部分
                new_mask[:, :old_len, :old_len] = self._cached_mask
                
                # 只填充新增的行
                for i in range(old_len, src_len):
                    if i < prefix_len:
                        # Prefix token: 因果mask
                        new_mask[:, i, :i + 1] = True
                    else:
                        # Tree token
                        tree_pos = i - prefix_len
                        if prefix_len > 0:
                            new_mask[:, i, :prefix_len] = True
                        new_mask[:, i, prefix_len:src_len] = tree_template[:, tree_pos, :]
                
                self._cached_mask = new_mask
                self._cached_mask_src_len = src_len
            else:
                # 创建全新的mask
                mask = torch.zeros(B, src_len, src_len, dtype=torch.bool, device=device)
                
                # Prefix部分
                if prefix_len > 0:
                    mask[:, :prefix_len, :prefix_len] = torch.tril(
                        torch.ones(prefix_len, prefix_len, dtype=torch.bool, device=device)
                    )
                
                # Tree部分
                if tree_len > 0 and prefix_len < src_len:
                    # Tree tokens可以看到所有prefix
                    mask[:, prefix_len:, :prefix_len] = True
                    # Tree内部可见性
                    mask[:, prefix_len:, prefix_len:] = tree_template
                
                self._cached_mask = mask
                self._cached_mask_src_len = src_len
        
        # 返回当前需要的行
        return self._cached_mask[:, -current_token_count:, :]
    
    def _get_or_create_tree_template(self, tree_attention_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        """获取或创建tree mask模板"""
        with self._profile_section("get_template_total"):
            # 计算cache key
            with self._profile_section("compute_cache_key"):
                cache_key = (
                    tree_attention_mask.shape,
                    hash(tree_attention_mask.cpu().numpy().tobytes())
                )
            
            # 检查缓存
            if cache_key in self._tree_mask_template_cache:
                with self._profile_section("cache_hit"):
                    template = self._tree_mask_template_cache[cache_key]
                    return template.to(device) if template.device != device else template
            
            # Cache miss - 需要解包
            with self._profile_section("cache_miss_unpack"):
                # 解包tree mask
                if hasattr(torch, "unpackbits"):
                    with self._profile_section("torch_unpackbits"):
                        bits = torch.unpackbits(tree_attention_mask.to(device), dim=-1)
                else:
                    with self._profile_section("fallback_unpackbits"):
                        bits = self._unpackbits_fallback(tree_attention_mask.to(device), dim=-1)
                
                with self._profile_section("process_bits"):
                    bits = bits.flatten(start_dim=-3)
                    tree_len = bits.size(1)
                    tree_mask = bits[..., :tree_len].bool()
                
                # 缓存模板
                self._tree_mask_template_cache[cache_key] = tree_mask
                
                return tree_mask
    
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

    def _select_layer_past(self, cache_tensors: Sequence[torch.Tensor], prefix_length: int, 
                          kv_cache_position_ids: Optional[torch.Tensor] = None) -> Tuple[Sequence[torch.Tensor], Optional[torch.Tensor]]:
        """Extract first {prefix_length} tokens and optionally specific positions based on kv_cache_position_ids"""
        with self._profile_section("select_past_total"):
            key_cache, value_cache = list(cache_tensors[0::2]), list(cache_tensors[1::2])
            new_position_mapping = None
            
            for i in range(len(key_cache)):
                with self._profile_section(f"process_layer_{i}"):
                    # Flatten cache
                    with self._profile_section("flatten_cache"):
                        key_cache[i] = key_cache[i].flatten(0, 1)
                        value_cache[i] = value_cache[i].flatten(0, 1)
                    
                    k, v = key_cache[i], value_cache[i]
                    
                    if kv_cache_position_ids is not None and not is_dummy(kv_cache_position_ids):
                        with self._profile_section("select_with_position_ids"):
                            logger.info(f"Selecting KV cache using position_ids: {kv_cache_position_ids}")
                            
                            position_ids = kv_cache_position_ids.to(k.device)
                            
                            if position_ids.dim() == 1:
                                position_ids = position_ids.unsqueeze(0)
                            
                            batch_size = position_ids.shape[0]
                            num_tree_positions = position_ids.shape[1]
                            num_kv_heads = k.shape[0] // batch_size
                            
                            all_positions_list = []
                            continuous_positions_list = []
                            
                            for batch_idx in range(batch_size):
                                batch_positions = position_ids[batch_idx]
                                root_position = batch_positions[0].item()
                                prefix_positions = torch.arange(0, root_position, device=position_ids.device)
                                complete_positions = torch.cat([prefix_positions, batch_positions])
                                logger.info(f"_select_layer_past, complete_positions: {complete_positions}")
                                continuous_positions = torch.arange(len(complete_positions), device=position_ids.device)
                                all_positions_list.append(complete_positions)
                                continuous_positions_list.append(continuous_positions)
                            
                            max_length = max(len(pos) for pos in all_positions_list)
                            padded_positions = torch.zeros(batch_size, max_length, dtype=torch.long, device=position_ids.device)
                            position_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=position_ids.device)
                            
                            for batch_idx, positions in enumerate(all_positions_list):
                                seq_len = len(positions)
                                padded_positions[batch_idx, :seq_len] = positions
                                position_mask[batch_idx, :seq_len] = True
                            
                            padded_positions = torch.clamp(padded_positions, 0, k.shape[2] - 1)
                            expanded_positions = padded_positions.repeat_interleave(num_kv_heads, dim=0)
                            expanded_mask = position_mask.repeat_interleave(num_kv_heads, dim=0)
                            
                            selected_key = torch.gather(k, 2, expanded_positions.unsqueeze(1).expand(-1, k.shape[1], -1))
                            selected_value = torch.gather(v, 1, expanded_positions.unsqueeze(2).expand(-1, -1, v.shape[2]))
                            
                            if expanded_mask.any():
                                mask_key = expanded_mask.unsqueeze(1).expand(-1, k.shape[1], -1)
                                mask_value = expanded_mask.unsqueeze(2).expand(-1, -1, v.shape[2])
                                selected_key = selected_key * mask_key.to(selected_key.dtype)
                                selected_value = selected_value * mask_value.to(selected_value.dtype)
                            
                            valid_length = max(len(pos) for pos in all_positions_list)
                            selected_key = selected_key[:, :, :valid_length]
                            selected_value = selected_value[:, :valid_length, :]
                            
                            key_cache[i] = selected_key
                            value_cache[i] = selected_value
                            
                            if i == 0:
                                new_position_mapping = torch.arange(valid_length, device=position_ids.device)
                            
                            logger.info(
                                f"[KV] L{i:02d} selected key shape {tuple(selected_key.shape)} "
                                f"value shape {tuple(selected_value.shape)} "
                                f"root_positions: {[pos[0].item() for pos in all_positions_list]} "
                                f"total_positions: {[len(pos) for pos in all_positions_list]} "
                                f"compacted to continuous length: {valid_length}"
                            )
                    else:
                        with self._profile_section("select_prefix_only"):
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
            
            with self._profile_section("create_layer_past"):
                layer_past = tuple(chain(*zip(key_cache, value_cache)))
                logger.info(f"cache_tensors size: {len(cache_tensors)}, selected layer_past size: {len(layer_past)}")
                result = PerDeviceTensors(*layer_past) if len(self.module.module_shards) > 1 else layer_past
            
            return result, new_position_mapping

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