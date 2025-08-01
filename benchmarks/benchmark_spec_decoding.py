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
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # Using use_fast=False since LlamaTokenizerFast takes a long time to start, and we decode 1 token at a time anyway
    
    ssm = AutoModelForCausalLM.from_pretrained("JackFram/llama-68m")

    model = AutoDistributedSpeculativeModel.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    )
    logger.info(f"Created model: {process_idx=} {model.device=}")

    result = ""
    step_times = []
    start_time = perf_counter()
    # with model.transformer.h.inference_session(max_length=args.seq_len) as sess:
    #     for step in range(args.seq_len):
    #         start_time = perf_counter()

    #         outputs = model.generate(max_new_tokens=1, session=sess)
    #         result += tokenizer.decode(outputs[0])

    #         logger.info(f"benchmark_inference, result: {result}")
    #         if step >= args.warmup_steps:
    #             step_times.append(perf_counter() - start_time)
    #             speed = 1 / np.mean(step_times)
    #             logger.info(f"{process_idx=} {step=} {speed=:.2f}")
    
    test_prompt = "Hello world from Xu, I am a master student from"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt", add_special_tokens=False)
    result = model.generate(input_ids=input_ids, ssm=ssm)
    time = perf_counter() - start_time
    generated_tokens_num = result.shape[1] - input_ids.shape[1]
    speed = generated_tokens_num / time
    logger.info(f"benchmark_inference, result: {result}, generated_tokens_num: {generated_tokens_num}, time: {time} speed: {speed}")
    result_pipe.send(speed)


if __name__ == "__main__":
    main()
