#!/bin/bash

EXEC=./bin/expint_exec
SANITIZER=/usr/local/cuda-12.8/bin/compute-sanitizer
LOGDIR=logs
mkdir -p $LOGDIR

echo "[Benchmark] Running square performance benchmarks..."

for size in 5000 8192 16384 20000; do
    echo "[+] Running: -n $size -m $size"
    $EXEC -n $size -m $size -t > $LOGDIR/shared_gpu_n${size}_m${size}.txt
    $EXEC -n $size -m $size -g -t > $LOGDIR/shared_cpu_n${size}_m${size}.txt
done

echo
echo "[Sanitizer] Checking non-square configurations with compute-sanitizer..."

# 非方阵1
echo "[+] Sanitizing: -n 5000 -m 6000"
$SANITIZER $EXEC -n 5000 -m 6000 -c -t > $LOGDIR/sanitizer_gpu_n5000_m6000.txt

# 非方阵2
echo "[+] Sanitizing: -n 8192 -m 4096"
$SANITIZER $EXEC -n 8192 -m 4096 -c -t > $LOGDIR/sanitizer_gpu_n8192_m4096.txt

echo
echo "[Done] All benchmarks and memory checks completed. Logs saved in $LOGDIR/"
