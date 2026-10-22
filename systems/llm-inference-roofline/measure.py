import argparse
import json
import os
import random
import statistics
import time

MODEL_ID="Qwen/Qwen3-8B"
GRID=[
    (1,512),(8,512),(32,512),(64,512),(128,512),
    (1,2048),(8,2048),(32,2048),(64,2048),(128,2048)
]

_llm=None


def load_model():
    global _llm
    if _llm is None:
        from vllm import LLM
        _llm=LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_seqs=256,
            max_num_batched_tokens=32768,
            enable_prefix_caching=False
        )
    return _llm


def generate(batch,prompt_len,max_tokens):
    from vllm import SamplingParams
    llm=load_model()
    prompts=[]
    for _ in range(batch):
        tokens=[random.randint(1000,100000) for _ in range(prompt_len)]
        prompts.append({"prompt_token_ids":tokens})
    params=SamplingParams(temperature=0,max_tokens=max_tokens,ignore_eos=True)
    start=time.perf_counter()
    llm.generate(prompts,params,use_tqdm=False)
    return time.perf_counter()-start


def measure_point(batch,input_len,output_len=128,repeats=3):
    generate(batch,input_len,1)
    generate(batch,input_len,output_len)
    first_token=statistics.median(generate(batch,input_len,1) for _ in range(repeats))
    all_tokens=statistics.median(generate(batch,input_len,output_len) for _ in range(repeats))
    tpot=(all_tokens-first_token)/(output_len-1)
    return {
        "batch":batch,
        "input_len":input_len,
        "output_len":output_len,
        "ttft_ms":round(first_token*1e3,2),
        "tpot_ms":round(tpot*1e3,2),
        "tokens_per_s":round(batch/tpot)
    }


def gpu_name():
    import torch
    return torch.cuda.get_device_name(0)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-len",type=int,default=128)
    parser.add_argument("--out",default=None)
    args=parser.parse_args()

    rows=[]
    for batch,input_len in GRID:
        result=measure_point(batch,input_len,args.output_len)
        rows.append(result)
        print(result)

    name=gpu_name()
    output_path=args.out or f"results/{name.replace(' ','_')}.json"
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    with open(output_path,"w") as file:
        json.dump({"gpu":name,"model":MODEL_ID,"measurements":rows},file,indent=2)
    print("saved",output_path)
