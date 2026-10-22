import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import roofline


def main(path):
    with open(path) as file:
        data=json.load(file)
    gpu=roofline.gpu_key(data["gpu"])
    rows=[]
    for measured in data["measurements"]:
        predicted=roofline.predict(gpu,measured["batch"],measured["input_len"],measured["output_len"])
        rows.append({
            "batch":measured["batch"],
            "input_len":measured["input_len"],
            "ttft_meas":measured["ttft_ms"],
            "ttft_pred":predicted["ttft_ms"],
            "ttft_eff":round(100*predicted["ttft_ms"]/measured["ttft_ms"]),
            "tpot_meas":measured["tpot_ms"],
            "tpot_pred":predicted["tpot_ms"],
            "tpot_eff":round(100*predicted["tpot_ms"]/measured["tpot_ms"]),
            "decode_bound":predicted["decode_bound"]
        })

    header="| batch | prompt | TTFT measured (ms) | TTFT roofline (ms) | efficiency | TPOT measured (ms) | TPOT roofline (ms) | efficiency | decode limited by |"
    lines=[f"Qwen3-8B on {data['gpu']} (roofline numbers for {gpu})","",header,"|"+"---|"*9]
    for row in rows:
        values=[
            row["batch"],row["input_len"],row["ttft_meas"],row["ttft_pred"],
            str(row["ttft_eff"])+"%",row["tpot_meas"],row["tpot_pred"],
            str(row["tpot_eff"])+"%",row["decode_bound"]
        ]
        lines.append("| "+" | ".join(str(value) for value in values)+" |")
    table="\n".join(lines)
    print(table)

    output_dir=os.path.dirname(path)
    with open(os.path.join(output_dir,"compare.md"),"w") as file:
        file.write(table+"\n")

    fig,ax=plt.subplots(figsize=(6,4))
    for input_len,color in [(512,"tab:blue"),(2048,"tab:orange")]:
        selected=[row for row in rows if row["input_len"]==input_len]
        if not selected:
            continue
        batches=[row["batch"] for row in selected]
        ax.plot(batches,[row["tpot_meas"] for row in selected],"o-",color=color,label=f"measured, {input_len}-token prompts")
        ax.plot(batches,[row["tpot_pred"] for row in selected],"--",color=color,label=f"roofline, {input_len}-token prompts")
    ax.set_xscale("log",base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("time per output token (ms)")
    ax.set_title(f"Qwen3-8B decode on {gpu}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,"compare.png"),dpi=130)
    print("saved",os.path.join(output_dir,"compare.md"),"and compare.png")


if __name__=="__main__":
    main(sys.argv[1])
