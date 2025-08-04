#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter

import numpy as np
import torch
from hivemind.utils.logging import get_logger
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM, AutoDistributedSpeculativeModel
from petals.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS

from transformers import AutoTokenizer, AutoModelForCausalLM

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS, help="Initial peers")
    parser.add_argument("--torch_dtype", type=str, default="float32", help="Torch dtype")
    parser.add_argument("--n_processes", type=str, default=1, help="Number of concurrent processes")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Number of warmup steps")
    args = parser.parse_args()

    if args.n_processes == "n_gpus":
        args.n_processes = torch.cuda.device_count()
    else:
        args.n_processes = int(args.n_processes)

    pipe_recv, pipe_send = mp.Pipe(duplex=False)
    processes = [mp.Process(target=benchmark_inference, args=(i, args, pipe_send)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    speed = np.mean([pipe_recv.recv() for _ in range(args.n_processes)])
    logger.info(f"Final result: {speed=:.2f}")


@torch.inference_mode()
def benchmark_inference(process_idx, args, result_pipe):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    test_prompt = "Hello world from Xu, I am a master student from"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    # Using use_fast=False since LlamaTokenizerFast takes a long time to start, and we decode 1 token at a time anyway
    
    ssm = AutoModelForCausalLM.from_pretrained("JackFram/llama-68m")
    
    
    ssm = ssm.to(device).eval()
    # ssm = ssm.to(input_ids.device).eval().half()
    # ssm = torch.compile(ssm)
    
    logger.info(f"input_ids.device: {input_ids.device}")
    logger.info(f"ssm device: {next(ssm.parameters()).device}")
    logger.info(f"ssm dtype: {next(ssm.parameters()).dtype}")

    model = AutoDistributedSpeculativeModel.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    ).to(device)
    logger.info(f"Created model: {process_idx=} {model.device=}")

    result = ""
    step_times = []
    
    with torch.no_grad():
        dummy_input = torch.ones(1, 8, dtype=torch.long, device=device)
        ssm(dummy_input, attention_mask=torch.ones_like(dummy_input))
    
    start_time = perf_counter()
    result = model.generate(input_ids=input_ids, ssm=ssm)
    time = perf_counter() - start_time
    generated_tokens_num = result.shape[1] - input_ids.shape[1]
    speed = generated_tokens_num / time
    decoded_result = tokenizer.decode(result[0], skip_special_tokens=True)
    
    logger.info(f"benchmark_inference, result: {result}, generated_tokens_num: {generated_tokens_num}, time: {time} speed: {speed}, decoded_result: {decoded_result}")
    result_pipe.send(speed)


if __name__ == "__main__":
    main()
