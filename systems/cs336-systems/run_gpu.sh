#!/bin/bash
# Runs everything on a RunPod pod with 2x H100:   bash run_gpu.sh
# Results go to results/*.csv, profiles to results/profiles/*.nsys-rep.
# A failed step is reported and the script moves on to the next one.

cd "$(dirname "$0")"
export PYTHONPATH=$PWD:$PWD/cs336-basics
# cu128 works with older drivers, PyPI's default torch 2.11 needs a CUDA 13 driver
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

echo "== setup"
mkdir -p results/profiles
rm -f results/*.csv
nvidia-smi | tee results/gpu_info.txt
nvidia-smi topo -m | tee -a results/gpu_info.txt
pip install "torch==2.11.*" --index-url $TORCH_INDEX
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); assert torch.cuda.device_count() >= 2" || { echo "need torch with 2 GPUs"; exit 1; }

if ! command -v nsys > /dev/null; then
    echo "== installing nsys"
    . /etc/os-release
    repo=https://developer.download.nvidia.com/devtools/repos/ubuntu${VERSION_ID//./}/$(dpkg --print-architecture)
    apt-get update -y && apt-get install -y --no-install-recommends wget gnupg
    wget -qO - $repo/nvidia.pub | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] $repo/ /" > /etc/apt/sources.list.d/nvidia-devtools.list
    apt-get update -y && apt-get install -y nsight-systems-cli
fi
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems*/*/bin/nsys 2> /dev/null | tail -1)

echo "== Stanford tests: DDP, sharded optimizer and FSDP on CPU (gloo)"
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_ddp.py tests/test_sharded_optimizer.py tests/test_fsdp.py 2>&1 | tee results/tests_cpu.txt

echo "== Stanford tests: attention, Triton runs on the GPU"
python -m pytest -q -rs tests/test_attention.py 2>&1 | tee results/tests_gpu.txt

echo "== single GPU benchmarks"
for size in small medium large xl; do
    python benchmark.py --size $size
    python benchmark.py --size $size --bf16
done
python benchmark.py --size small --warmup 0

echo "== attention benchmark"
timeout 20m python attention_bench.py

echo "== DDP, sharded optimizer and FSDP on 2 GPUs"
for mode in naive flat overlap sharded fsdp; do
    timeout 15m torchrun --nproc_per_node=2 ddp_bench.py --mode $mode --size xl
done

echo "== nsys profiles, naive vs overlapped DDP"
if [ -n "$NSYS" ]; then
    for mode in naive overlap; do
        timeout 15m $NSYS profile --trace cuda,nvtx --sample none --cpuctxsw none --force-overwrite true \
            -o results/profiles/ddp_$mode \
            torchrun --nproc_per_node=2 ddp_bench.py --mode $mode --size xl --warmup 2 --steps 3 --csv results/profiles/profile_runs.csv
    done
else
    echo "nsys not found, skipping profiles"
fi

echo "== tables"
python make_tables.py | tee results/tables.md
echo "done in $((SECONDS / 60)) min"
