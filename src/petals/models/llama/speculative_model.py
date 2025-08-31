from typing import Optional, Union, List, Tuple, Any
import torch
import numpy as np
import contextlib
from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput, GenerationMixin
from transformers.models.llama import LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

from petals.models.llama.config import DistributedLlamaConfig
from petals.models.llama.model import DistributedLlamaForCausalLM
from petals.models.llama.spe_dec_tree import SpeculativeTree, TreeNode, prepare_incremental_tree_batch

from petals.client.remote_generation import RemotePastKeyValues
from petals.client.inference_session import InferenceSession

from transformers import AutoTokenizer, AutoModelForCausalLM

from hivemind.utils.logging import get_logger

import time

logger = get_logger()

class DistributedLlamaForSpeculativeGeneration(DistributedLlamaForCausalLM):
    def __init__(self, config: DistributedLlamaConfig):
        super().__init__(config)
        
    def generate(
        self,
        input_ids: torch.LongTensor,
        ssm: LlamaForCausalLM,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        streamer: Optional["BaseStreamer"] = None,
        beam_width: int = 2,
        max_tree_depth: int = 4,
        use_kv_cache: bool = True,
        kv_cache_window: int = 2048,
        max_new_tokens: int = 128,
        **model_kwargs,
    ) -> torch.LongTensor:
        
        generation_config = generation_config or getattr(self, "generation_config", GenerationConfig())
        logits_processor = logits_processor or LogitsProcessorList()
        stopping_criteria = stopping_criteria or StoppingCriteriaList()

        generation_config.do_sample = False
        generation_config.return_dict_in_generate = False

        # Calculate session max length - this is critical for distributed inference
        session_max_length = 1280

        # Use inference session for proper distributed caching
        with self.transformer.h.inference_session(max_length=session_max_length) as session:
            return self._sample_with_session(
                input_ids=input_ids,
                ssm=ssm,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                session=session,
                streamer=streamer,
                beam_width=beam_width,
                max_tree_depth=max_tree_depth,
                use_kv_cache=use_kv_cache,
                kv_cache_window=kv_cache_window,
                max_new_tokens=max_new_tokens,
                **model_kwargs,
            )
        
    def _sample_with_session(
        self,
        input_ids: torch.LongTensor,
        ssm: LlamaForCausalLM,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        session: InferenceSession,
        streamer: Optional["BaseStreamer"],
        beam_width: int = 2,
        max_tree_depth: int = 3,
        use_kv_cache: bool = True,
        kv_cache_window: int = 2048,
        max_new_tokens: int = 128,
        **model_kwargs,
    ) -> torch.LongTensor:
        logger.info("Starting speculative decoding with distributed inference session!")
        # tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=False)
        # start_time_total = time.time()
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        finished = False
        
        # Initialize past_key_values for session tracking
        past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(session.position)
        
        is_first_iteration = True
        step_idx = 0
        current_input_ids = input_ids
        result = ""
        llm_generated_token = None
        
        # total_steps = 0
        # total_tree_build_time = 0
        # total_verify_time = 0
        # total_verify_tokens_num = 0
        # total_concat_time = 0
        
        while not finished and current_input_ids.shape[1] < input_ids.shape[1] + max_new_tokens:
            # logger.info(f"\n==================== STEP {step_idx} ====================")
            # logger.info(f"[DEBUG] Current sequence length: {current_input_ids.shape[1]}")
            # logger.info(f"[DEBUG] Session position: {session.position}")
            # step_start = time.time()
            # t1 = time.time()
            # 1. Build speculative trees using SSM
            spec_trees = self._build_speculative_trees_batched(
                current_input_ids, ssm, beam_width, max_tree_depth
            )
            # t2 = time.time()
            
            # 2. Verify trees using distributed inference - but through forward() call
            verified_tokens, verified_tokens_positions, past_key_values, llm_generated_token = self._verify_trees_with_forward(
                input_ids=current_input_ids,
                llm_generated_token=llm_generated_token,
                trees=spec_trees,
                logits_processor=logits_processor,
                past_key_values=past_key_values,
                is_first_iteration=is_first_iteration,
                use_kv_cache=use_kv_cache,
                kv_cache_window=kv_cache_window
            )
            # t3 = time.time()
            
            past_key_values.set_kv_cache(verified_tokens_positions)
            
            is_first_iteration = False
            
            # 3. Apply stopping conditions
            if has_eos_stopping_criteria:
                verified_tokens = verified_tokens * unfinished_sequences + generation_config.pad_token_id * (
                    1 - unfinished_sequences
                )

            # 4. Update input sequence
            if verified_tokens is not None:
                # total_verify_tokens_num += verified_tokens.shape[1]
                current_input_ids = torch.cat([current_input_ids, verified_tokens], dim=-1)
                current_input_ids = torch.cat([current_input_ids, llm_generated_token.unsqueeze(0)], dim=-1)
                # logger.info(f"[DEBUG] Verified tokens appended: {verified_tokens.tolist()}, llm_generated_token: {llm_generated_token}")
                # logger.info(f"[DEBUG] New sequence length: {current_input_ids.shape[1]}")
                # new_added_text = ""
                # for i in range(len(verified_tokens)):
                #     new_added_text += tokenizer.decode(verified_tokens[i])
                # new_added_text += " " + tokenizer.decode(llm_generated_token)
                # logger.info(f"[DEBUG] new added: {new_added_text}")
                # result = ""
                # for i in range(len(current_input_ids)):
                #     result += tokenizer.decode(current_input_ids[i])
                # logger.info(f"[DEBUG] temp result: {result}")

                if streamer is not None:
                    streamer.put(verified_tokens.cpu())
            else :
                current_input_ids = torch.cat([current_input_ids, llm_generated_token.unsqueeze(0)], dim=-1)
                # logger.info(f"[DEBUG] new added llm_generated_token: {llm_generated_token}, text: {tokenizer.decode(llm_generated_token)}")
                # result = ""
                # for i in range(len(current_input_ids)):
                #     result += tokenizer.decode(current_input_ids[i])
                # logger.info(f"[DEBUG] temp result: {result}")
            
            # t4 = time.time()

            # 5. Check if finished
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(current_input_ids, None)
            finished = unfinished_sequences.max() == 0
            step_idx += 1
            # logger.info(f"finished: {finished}, current_input_ids.shape[1]: {current_input_ids.shape[1]}, input_ids.shape[1]: {input_ids.shape[1]}, max_new_tokens: {max_new_tokens}")
            
            # total_steps += 1
            # total_tree_build_time += t2 - t1
            # total_verify_time += t3 - t2
            # total_concat_time += t4 - t3

        if streamer is not None:
            streamer.end()
        
        # end_time_total = time.time()
        # logger.info(f"[Timing] _sample_with_session total: {end_time_total - start_time_total:.4f}s | "
        #         f"steps: {total_steps} | "
        #         f"tree_build: {total_tree_build_time:.4f}s | "
        #         f"verify: {total_verify_time:.4f}s | "
        #         f"concat_and_stream: {total_concat_time:.4f}s | "
        #         f"total_verify_tokens_num: {total_verify_tokens_num}")
        return current_input_ids
    
    def _verify_trees_with_forward(
        self,
        input_ids: torch.LongTensor,
        llm_generated_token: torch.Tensor,
        trees: List[SpeculativeTree],
        logits_processor: LogitsProcessorList,
        past_key_values: RemotePastKeyValues,
        is_first_iteration: bool,
        use_kv_cache: bool,
        kv_cache_window: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor, RemotePastKeyValues, torch.Tensor]:
        """
        Verify speculative trees using standard forward() call within the active session context
        """
        
        tree_tokens, attention_mask, batch_node_paths = prepare_incremental_tree_batch(
            trees, input_ids, input_ids.device
        )
        
        # logger.info(f"[DEBUG] Tree tokens shape: {tree_tokens.shape}")
        # logger.info(f"[DEBUG] Tree tokens: {tree_tokens}")
        # logger.info(f"[DEBUG] Active session position: {self.transformer.h.active_session.position if self.transformer.h.active_session else 'None'}")
        
        if attention_mask is None or tree_tokens.shape[1] == 0:
            logger.warning("No tree tokens to verify, falling back to regular generation")
            return self._fallback_generation_with_forward(input_ids, logits_processor, past_key_values), past_key_values
        
        tree_mask_packed = self.pack_bool_mask_to_int64(attention_mask)
        
        with torch.no_grad():
            if not use_kv_cache:
                # No cache: process tree tokens directly
                logger.warning("Processing without KV cache, may cause error!!!")
                outputs = self(
                    input_ids=tree_tokens,
                    attention_mask=tree_mask_packed,
                    past_key_values=None,
                    use_cache=False
                )
                logits = outputs.logits
                new_past_key_values = past_key_values
                
            elif is_first_iteration or past_key_values is None:
                # First iteration: process full sequence to establish cache
                full_sequence = torch.cat([input_ids, tree_tokens], dim=-1)
                # logger.info(f"[DEBUG] First iteration - processing full sequence of length: {full_sequence.shape[1]}")
                
                outputs = self(
                    input_ids=full_sequence,
                    attention_mask=tree_mask_packed,  # Let the session handle attention
                    past_key_values=None,  # Start fresh
                    use_cache=True
                )
                
                # Extract only the tree portion of the logits
                logits = outputs.logits
                
                # Update past_key_values tracking
                if past_key_values is None:
                    new_past_key_values = RemotePastKeyValues()
                else:
                    new_past_key_values = past_key_values
                
                # The session will automatically handle the KV cache positioning
                if self.transformer.h.active_session:
                    new_past_key_values.update_seen(self.transformer.h.active_session.position)
                
                # logger.info(f"[DEBUG] First iteration completed, session position: {self.transformer.h.active_session.position if self.transformer.h.active_session else 'None'}")
                
            else:
                # Subsequent iterations: use existing cache
                active_session = self.transformer.h.active_session
                if active_session is None:
                    raise ValueError("No active session available for cached inference")
                
                # Handle cache window management
                if active_session.position > kv_cache_window:
                    trim_amount = active_session.position - kv_cache_window
                    active_session.position = kv_cache_window
                    # logger.info(f"Trimmed cache: reset position from {active_session.position + trim_amount} to {kv_cache_window}")
                
                # logger.info(f"[DEBUG] Subsequent iteration - processing tree tokens of length: {tree_tokens.shape[1]}")
                if llm_generated_token is None:
                    full_sequence = tree_tokens
                else:
                    full_sequence = torch.cat([llm_generated_token.unsqueeze(0), tree_tokens], dim=-1)
                
                # Process tree tokens with existing cache
                outputs = self(
                    input_ids=full_sequence,
                    attention_mask=tree_mask_packed,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits
                new_past_key_values = past_key_values
                new_past_key_values.update_seen(active_session.position)
                
                # logger.info(f"[DEBUG] Subsequent iteration completed, session position: {active_session.position}")
        
        # Extract verification results
        verified_tokens, verified_tokens_positions, llm_generated_token = self._extract_best_verified_paths_fixed(
            logits, batch_node_paths, input_ids, logits_processor, tree_tokens.shape[1]
        )
        # logger.info(f"[DEBUG] Verified tokens (per batch): {verified_tokens.tolist() if verified_tokens is not None else None}, verified_tokens_positions: {verified_tokens_positions}, llm_generated_token: {llm_generated_token}")
        return verified_tokens, verified_tokens_positions, new_past_key_values, llm_generated_token
    
    def pack_bool_mask_to_int64(self, mask_bool: torch.Tensor) -> torch.Tensor:
        batch_size, n, m = mask_bool.shape
        assert n == m, "Must be square matrix"
        
        # 转换为numpy uint8
        mask_np = mask_bool.cpu().numpy().astype(np.uint8)
        
        # 按行打包 - packbits会将每行的n个bool值打包成ceil(n/8)个字节
        # axis=-1 表示对最后一维（每行）进行打包
        packed = np.packbits(mask_np, axis=-1)
        
        # packed现在的shape是 (batch_size, n, ceil(n/8))
        # 需要padding到8的倍数（为了形成int64）
        bytes_per_row = packed.shape[-1]
        padded_bytes = ((bytes_per_row + 7) // 8) * 8  # 向上取整到8的倍数
        
        if padded_bytes > bytes_per_row:
            pad_width = [(0, 0), (0, 0), (0, padded_bytes - bytes_per_row)]
            packed = np.pad(packed, pad_width, constant_values=0)
        
        # 重塑为 (batch_size, n, num_int64, 8)
        num_int64 = padded_bytes // 8
        packed = packed.reshape(batch_size, n, num_int64, 8)
        
        return torch.from_numpy(packed).to(mask_bool.device)
    
    def _fallback_generation_with_forward(
        self, 
        input_ids: torch.LongTensor, 
        logits_processor: LogitsProcessorList,
        past_key_values: RemotePastKeyValues,
        temperature: float = 1.0
    ) -> torch.LongTensor:
        """
        Fallback to regular generation using forward() call within active session
        """
        try:
            logger.info("[DEBUG] Using fallback generation")
            
            # Generate single token using standard forward call
            outputs = self(
                input_ids=input_ids[:, -1:],  # Just the last token
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :]  # Last position logits
            
            # Apply logits processors
            processed_logits = logits
            for processor in logits_processor:
                processed_logits = processor(input_ids, processed_logits)
            
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(processed_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(processed_logits, dim=-1, keepdim=True)
            
            # logger.info(f"[DEBUG] Fallback generated token: {next_token.tolist()}")
            return next_token
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            # Ultimate fallback - return EOS token
            eos_token_id = getattr(self.config, 'eos_token_id', 2)
            return torch.tensor([[eos_token_id]], device=input_ids.device)
    
    # Keep your existing methods with minimal changes
    def _build_speculative_trees_batched(
        self, 
        input_ids: torch.LongTensor, 
        ssm: LlamaForCausalLM, 
        beam_width: int, 
        max_depth: int
    ) -> List[SpeculativeTree]:
        """Build speculative trees using the small model (SSM)"""
        # start_total = time.time()
        batch_size = input_ids.shape[0]
        trees = []

        for batch_idx in range(batch_size):
            # start_batch = time.time()
            root_token = input_ids[batch_idx, -1].item()
            tree = SpeculativeTree(root_token, f"req_{batch_idx}")
            # logger.info(f"[Tree] Building tree for batch {batch_idx}, root_token: {root_token}")

            for depth in range(max_depth):
                # start_depth = time.time()

                current_nodes = tree.get_nodes_at_depth(depth)
                if not current_nodes:
                    break

                # Build contexts
                contexts = []
                for node in current_nodes:
                    path_to_node = node.get_path_from_root()
                    context = torch.cat([
                        input_ids[batch_idx, :-1],
                        torch.tensor([root_token] + path_to_node, device=input_ids.device)
                    ])
                    contexts.append(context)

                if not contexts:
                    break

                max_len = max(len(ctx) for ctx in contexts)
                padded_contexts = []
                attention_masks = []

                for ctx in contexts:
                    pad_len = max_len - len(ctx)
                    pad_token_id = getattr(ssm.config, 'pad_token_id', 0)

                    padded = torch.cat([
                        torch.full((pad_len,), pad_token_id, dtype=torch.long, device=input_ids.device),
                        ctx
                    ])

                    mask = torch.cat([
                        torch.zeros(pad_len, dtype=torch.bool, device=input_ids.device),
                        torch.ones(len(ctx), dtype=torch.bool, device=input_ids.device)
                    ])

                    padded_contexts.append(padded)
                    attention_masks.append(mask)

                batch_contexts = torch.stack(padded_contexts)
                batch_masks = torch.stack(attention_masks)

                # SSM forward
                # start_ssm = time.time()
                with torch.no_grad():
                    outputs = ssm(batch_contexts, attention_mask=batch_masks)
                    batch_logits = outputs.logits[:, -1, :]
                # end_ssm = time.time()

                # Generate candidates
                # start_add_layer = time.time()
                candidates_per_node = []
                for i in range(len(current_nodes)):
                    logits = batch_logits[i]
                    top_k_values, top_k_indices = torch.topk(logits, k=beam_width)
                    probs = torch.softmax(logits, dim=-1)

                    candidates = []
                    for j in range(beam_width):
                        token_id = top_k_indices[j].item()
                        prob = probs[token_id].item()
                        candidates.append((token_id, prob))

                    candidates_per_node.append(candidates)

                try:
                    new_nodes = tree.add_layer(current_nodes, candidates_per_node)
                    if not new_nodes:
                        break
                except ValueError as e:
                    logger.warning(f"Failed to add tree layer: {e}")
                    break
                # end_add_layer = time.time()

                # logger.info(f"[Tree] Batch {batch_idx}, depth {depth}: "
                #             f"SSM infer: {end_ssm - start_ssm:.4f}s, "
                #             f"add_layer: {end_add_layer - start_add_layer:.4f}s"
                #             f"prepare context: {start_ssm-start_depth:.4f}")

            trees.append(tree)
            # end_batch = time.time()
            # logger.info(f"[Tree] Finished batch {batch_idx} in {end_batch - start_batch:.4f}s")

        # total_time = time.time() - start_total
        # logger.info(f"[Tree] Built {batch_size} speculative trees in {total_time:.4f}s")
        return trees
    
    def _extract_best_verified_paths_fixed(
        self,
        logits: torch.Tensor,
        batch_node_paths: List[List[List[TreeNode]]],
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        tree_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract best verified paths (simplified for batch=1)
        Returns: (verified_tokens, kv_cache_position_ids)
        
        kv_cache_position_ids的结构: 一维数组
        - 首个元素: 树根位置ID (input_ids.shape[1] - 1)
        - 后续元素: 与verified_tokens对应的位置ID
        """
        # 计算树的总长度
        total_tree_tokens = tree_len
        fallback_pos = max(0, logits.shape[1] - total_tree_tokens)
        
        # 树根的位置ID
        tree_root_position = input_ids.shape[1] - 1
        
        # 只处理第一个batch (batch=1)
        node_paths = batch_node_paths[0]
        best_verified = []
        best_positions = []
        best_score = -1
        
        # 找到最佳验证路径
        for node_path in node_paths:
            verified_tokens = []
            verified_positions = []
            
            for node in node_path:
                pos = node.position_in_sequence
                if pos >= logits.shape[1]:
                    break
                
                predicted_token = torch.argmax(logits[0, pos]).item()
                
                if predicted_token == node.token_id:
                    verified_tokens.append(node.token_id)
                    absolute_position = tree_root_position + node.position_in_sequence + 1
                    verified_positions.append(absolute_position)
                else:
                    break
            
            if len(verified_tokens) > best_score:
                best_score = len(verified_tokens)
                best_verified = verified_tokens
                best_positions = verified_positions
        
        # Handle empty verification case
        # logger.info(f"_extract_best_verified_paths_fixed, best_verified: {best_verified}, best_positions: {best_positions}, logits: {logits.shape}, fallback_pos: {fallback_pos}, input_ids.shape: {input_ids.shape}")
        if len(best_verified) == 0:
            # logger.warning("No tokens verified, using reverse indexing for fallback")
            
            # 生成fallback token
            final_logits = logits[0, fallback_pos-1:fallback_pos]  # 取单个位置的logits
            processed_logits = final_logits
            for processor in logits_processor:
                processed_logits = processor(input_ids, processed_logits)
            
            next_token = torch.argmax(processed_logits, dim=-1, keepdim=True)
            
            # 构建kv_cache_position_ids：[树根位置, fallback token位置]
            kv_cache_position_ids = torch.tensor([tree_root_position], 
                                            device=logits.device)
            
            # logger.info(f"[DEBUG] Fallback kv_cache_position_ids: {kv_cache_position_ids.tolist()}")
            llm_generated_token = torch.tensor([next_token], device=logits.device)
            return None, kv_cache_position_ids, llm_generated_token
        
        # 有验证token的情况 - 不需要填充，直接转tensor
        verified_tensor = torch.tensor([best_verified], device=logits.device)  # [1, num_verified]
        
        # Generate additional token using logits processor
        # if len(best_verified) < logits.shape[1]:
        #     pos = len(best_verified)
        # else:
        #     pos = logits.shape[1] - 1
        pos = best_positions[-1] - tree_root_position
        
        final_logits = logits[0, pos:pos+1]  # 取单个位置的logits
        
        # Apply logits processor
        processor_input = torch.cat([input_ids, verified_tensor], dim=1)
        processed_logits = final_logits
        for processor in logits_processor:
            processed_logits = processor(processor_input, processed_logits)
        
        next_token = torch.argmax(processed_logits, dim=-1, keepdim=True)
        
        # 计算下一个token的位置
        # if len(best_positions) > 0:
        #     next_pos = best_positions[-1] + 1
        # else:
        #     next_pos = input_ids.shape[1]
        
        # 合并verified tokens和新生成的token
        # verified_tensor = torch.cat([verified_tensor, next_token], dim=1)
        llm_generated_token = torch.tensor([next_token], device=logits.device)
        
        # 构建最终的kv_cache_position_ids：[树根位置, verified_positions..., next_token_position]
        all_positions = [tree_root_position] + best_positions
        kv_cache_position_ids = torch.tensor(all_positions, device=logits.device)
        
        # logger.info(f"[DEBUG] Final kv_cache_position_ids: {kv_cache_position_ids.tolist()}")
        
        return verified_tensor, kv_cache_position_ids, llm_generated_token