import argparse,json,os,re
import roofline,measure

for mod,fn in [(roofline,"predict"),(measure,"measure_point")]:
    if not hasattr(mod,fn):
        raise SystemExit(mod.__name__+".py has no function "+fn)

SYSTEM="""You help size LLM serving on one GPU.
1. Predict several batch sizes to locate the latency target.
2. Use predictions to choose GPU measurements. Never start measuring tiny batches far below the target.
3. Measure batches whose predicted TPOT is between 50% and 105% of the target.
4. Once you find an under-target batch and a borderline or over-target batch, use remaining measurements between them.
5. Use at most 4 GPU measurements. Borderline means within 2% of the target and is not a pass.
6. Report the largest tested batch safely under the target, any borderline batch, and the decode bottleneck.
Do not write a table. Python generates the table from tool results.
Only use numbers from tool results or the goal."""
def tool(name,desc):
    params={"batch":{"type":"integer"},"input_len":{"type":"integer"},"output_len":{"type":"integer"}}
    return {"type":"function","function":{"name":name,"description":desc,
            "parameters":{"type":"object","properties":params,"required":list(params)}}}

TOOLS=[tool("predict","Roofline prediction of TTFT and time per output token (TPOT) for one batch size."),
       tool("measure","Run vLLM on the GPU and measure TTFT and TPOT for one batch size. Also returns the roofline prediction for the same batch.")]
MAX_MEASUREMENTS=4
def detect_gpu():
    import torch
    name=torch.cuda.get_device_name(0)
    if hasattr(roofline,"gpu_key"):
        return roofline.gpu_key(name)
    if "H100" in name:
        return "H100-PCIe" if "PCIe" in name else "H100-SXM"
    if "A100" in name:
        return "A100-SXM"
    raise SystemExit("no roofline specs for "+name)
def vs_target(tpot,target):
    if tpot<0.98*target:
        return "under"
    if tpot<=1.02*target:
        return "borderline"
    return "over"
def run_tool(name,args,gpu,target,state):
    try:
        b,i,o=args["batch"],args["input_len"],args["output_len"]
    except KeyError as e:
        return {"error":"missing argument "+str(e)}
    p=roofline.predict(gpu,b,i,o)
    if name=="predict":
        return p
    if name!="measure":
        return {"error":"unknown tool "+name}
    if p["tpot_ms"]<target*0.5 or p["tpot_ms"]>target*1.05:
        return {"error":"Predicted TPOT is too far from target. Choose a batch closer to the latency boundary."}
    if not p["fits_in_memory"]:
        return {"error":"KV cache for this batch does not fit in GPU memory, vLLM would queue requests so TPOT would not be a real decode step"}
    if state["measured"]>=MAX_MEASUREMENTS:
        return {"error":"measurement budget of "+str(MAX_MEASUREMENTS)+" used up"}
    state["measured"]+=1
    m=measure.measure_point(b,i,o)
    m["predicted_tpot_ms"]=p["tpot_ms"]
    m["predicted_ttft_ms"]=p["ttft_ms"]
    m["efficiency_pct"]=round(100*p["tpot_ms"]/m["tpot_ms"])
    m["decode_bound"]=p["decode_bound"]
    m["vs_target"]=vs_target(m["tpot_ms"],target)
    return m
def numbers_in(x):
    if isinstance(x,bool):
        return []
    if isinstance(x,(int,float)):
        return [float(x)]
    if isinstance(x,dict):
        return [n for v in x.values() for n in numbers_in(v)]
    if isinstance(x,(list,tuple)):
        return [n for v in x for n in numbers_in(v)]
    if isinstance(x,str):
        return [float(s) for s in number_strings(x)]
    return []
def number_strings(text):
    text=re.sub(r"(?<=\d),(?=\d{3})","",text)
    text=re.sub(r"^\s*\d+\.\s","",text,flags=re.M)
    return re.findall(r"(?<![\w.-])\d+(?:\.\d+)?",text)

def unsupported_numbers(text,sources):
    allowed=[n for s in sources for n in numbers_in(s)]
    bad=[]
    for s in number_strings(text):
        x=float(s)
        decimals=len(s.split(".")[1]) if "." in s else 0
        tol=0.5*10**-decimals+1e-9
        if not any(abs(x-y)<=tol for y in allowed):
            bad.append(x)
    return bad
def run_agent(goal,gpu,model,target,client=None,max_turns=20):
    if client is None:
        from openai import OpenAI
        client=OpenAI()
    messages=[{"role":"system","content":SYSTEM},{"role":"user","content":goal}]
    sources=[goal,SYSTEM]
    log=[]
    state={"measured":0}
    retried=False
    for _ in range(max_turns):
        msg=client.chat.completions.create(model=model,messages=messages,tools=TOOLS).choices[0].message
        entry={"role":"assistant","content":msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"]=[{"id":c.id,"type":"function","function":{"name":c.function.name,"arguments":c.function.arguments}} for c in msg.tool_calls]
        messages.append(entry)
        if msg.tool_calls:
            for c in msg.tool_calls:
                args=json.loads(c.function.arguments)
                result=run_tool(c.function.name,args,gpu,target,state)
                print(c.function.name,args,"->",result)
                sources+=[args,result]
                log.append({"tool":c.function.name,"args":args,"result":result})
                messages.append({"role":"tool","tool_call_id":c.id,"content":json.dumps(result)})
            continue
        text=msg.content or ""
        bad=unsupported_numbers(text,sources)
        if bad and not retried:
            retried=True
            print("check failed, numbers not from any tool:",bad)
            messages.append({"role":"user","content":"These numbers do not appear in any tool result: "+str(bad)+". Rewrite using only numbers from tool results."})
            continue
        return {"conclusion":text,"unsupported_numbers":bad,"tool_calls":log}
    return {"conclusion":None,"unsupported_numbers":[],"tool_calls":log,"error":"ran out of turns"}
def results_table(log,target):
    rows=[c["result"] for c in log if c["tool"]=="measure" and "error" not in c["result"]]
    rows.sort(key=lambda r:r["batch"])
    lines=[f"| batch | roofline TPOT (ms) | measured TPOT (ms) | efficiency | vs {target:g} ms target |","|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['batch']} | {r['predicted_tpot_ms']} | {r['tpot_ms']} | {r['efficiency_pct']}% | {r['vs_target']} |")
    return "\n".join(lines)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--target-ms",type=float,default=20)
    ap.add_argument("--input-len",type=int,default=2048)
    ap.add_argument("--output-len",type=int,default=128)
    ap.add_argument("--model",default=os.environ.get("OPENAI_MODEL","gpt-4.1-mini"))
    ap.add_argument("--gpu",default=None)
    args=ap.parse_args()
    gpu=args.gpu or detect_gpu()
    target=args.target_ms
    goal=f"Find the largest batch size for Qwen3-8B on this GPU where time per output token stays under {target:g} ms, with {args.input_len}-token prompts and {args.output_len} output tokens."
    out=run_agent(goal,gpu,args.model,target)
    report="Goal: "+goal+"\n\n"+results_table(out["tool_calls"],target)+"\n\n"+(out["conclusion"] or out.get("error",""))+"\n"
    if out["unsupported_numbers"]:
        report+="\nWARNING: numbers not from any tool: "+str(out["unsupported_numbers"])+"\n"
    print("\n"+report)
    os.makedirs("results",exist_ok=True)
    with open("results/agent_report.md","w") as f:
        f.write(report)
    with open("results/agent_log.json","w") as f:
        json.dump(out,f,indent=2)
    print("saved results/agent_report.md and results/agent_log.json")
