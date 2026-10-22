import argparse

MODEL={
    "name":"Qwen3-8B",
    "hidden":4096,
    "layers":36,
    "heads":32,
    "kv_heads":8,
    "head_dim":128,
    "ffn":12288,
    "vocab":151936
}

GPUS={
    "H100-SXM":{"tflops":989,"bw_gbs":3350,"mem_gb":80},
    "H100-PCIe":{"tflops":756,"bw_gbs":2000,"mem_gb":80},
    "A100-SXM":{"tflops":312,"bw_gbs":2039,"mem_gb":80}
}

BYTES=2


def gpu_key(device_name):
    if "H100" in device_name:
        return "H100-PCIe" if "PCIe" in device_name else "H100-SXM"
    if "A100" in device_name:
        return "A100-SXM"
    raise ValueError(f"add {device_name} to GPUS first")


def layer_params(m=MODEL):
    hidden=m["hidden"]
    q=m["heads"]*m["head_dim"]
    kv=m["kv_heads"]*m["head_dim"]
    attention=hidden*q+2*hidden*kv+q*hidden
    mlp=3*hidden*m["ffn"]
    return m["layers"]*(attention+mlp)


def lm_head_params(m=MODEL):
    return m["vocab"]*m["hidden"]


def kv_bytes_per_token(m=MODEL):
    return 2*m["layers"]*m["kv_heads"]*m["head_dim"]*BYTES


def forward_pass(gpu,new_tokens,attn_pairs,kv_tokens,logit_tokens,m=MODEL):
    specs=GPUS[gpu]
    attention_flops=4*m["heads"]*m["head_dim"]*attn_pairs*m["layers"]
    flops=2*new_tokens*layer_params(m)+2*logit_tokens*lm_head_params(m)+attention_flops
    weight_bytes=(layer_params(m)+lm_head_params(m))*BYTES
    total_bytes=weight_bytes+kv_tokens*kv_bytes_per_token(m)
    compute_time=flops/(specs["tflops"]*1e12)
    memory_time=total_bytes/(specs["bw_gbs"]*1e9)
    bound="compute" if compute_time>memory_time else "memory"
    return max(compute_time,memory_time),bound


def prefill(gpu,batch,prompt_len):
    pairs=batch*prompt_len*(prompt_len+1)/2
    return forward_pass(gpu,batch*prompt_len,pairs,batch*prompt_len,batch)


def decode_step(gpu,batch,context):
    return forward_pass(gpu,batch,batch*(context+1),batch*(context+1),batch)


def fits_in_memory(gpu,batch,input_len,output_len,m=MODEL):
    weights=(layer_params(m)+2*lm_head_params(m))*BYTES
    kv=batch*(input_len+output_len)*kv_bytes_per_token(m)
    return weights+kv<=0.9*GPUS[gpu]["mem_gb"]*1e9


def predict(gpu,batch,input_len,output_len):
    ttft,prefill_bound=prefill(gpu,batch,input_len)
    tpot,decode_bound=decode_step(gpu,batch,input_len+output_len//2)
    return {
        "gpu":gpu,
        "batch":batch,
        "input_len":input_len,
        "output_len":output_len,
        "ttft_ms":round(ttft*1e3,2),
        "tpot_ms":round(tpot*1e3,2),
        "tokens_per_s":round(batch/tpot),
        "prefill_bound":prefill_bound,
        "decode_bound":decode_bound,
        "fits_in_memory":fits_in_memory(gpu,batch,input_len,output_len)
    }


def ridge_batch(gpu):
    specs=GPUS[gpu]
    return specs["tflops"]*1e12/(specs["bw_gbs"]*1e9)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--gpu",default="H100-SXM",choices=list(GPUS))
    parser.add_argument("--input-len",type=int,default=512)
    parser.add_argument("--output-len",type=int,default=128)
    args=parser.parse_args()

    weights_gb=(layer_params()+lm_head_params())*BYTES/1e9
    print(f"{MODEL['name']} on {args.gpu}")
    print(f"weights read per forward pass: {weights_gb:.1f} GB")
    print(f"ridge point: decode turns compute-bound around batch {ridge_batch(args.gpu):.0f}\n")
    print("batch | TTFT ms | TPOT ms | tokens/s | decode | fits")
    for batch in [1,8,32,64,128,256]:
        result=predict(args.gpu,batch,args.input_len,args.output_len)
        print(batch,result["ttft_ms"],result["tpot_ms"],result["tokens_per_s"],result["decode_bound"],result["fits_in_memory"],sep=" | ")
