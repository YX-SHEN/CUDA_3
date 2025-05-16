#!/bin/bash

mkdir -p logs

echo "====== [Running Shared Memory GPU Benchmarks] ======"
for size in 5000 8192 16384 20000; do
    ./bin/expint_exec -n $size -m $size -c -t > logs/shared_gpu_n${size}_m${size}.txt
done

echo "====== [Running CPU Benchmarks] ======"
for size in 5000 8192 16384 20000; do
    ./bin/expint_exec -n $size -m $size -g -t > logs/shared_cpu_n${size}_m${size}.txt
done

echo "====== [Running Non-Square System Sanity Test] ======"
# 非方阵兼容性测试（典型非对称测试点）
/usr/local/cuda-12.8/bin/compute-sanitizer ./bin/expint_exec -n 5000 -m 6000 -c -t > logs/sanitizer_gpu_n5000_m6000.txt
/usr/local/cuda-12.8/bin/compute-sanitizer ./bin/expint_exec -n 8192 -m 4096 -c -t > logs/sanitizer_gpu_n8192_m4096.txt

echo "====== [All Done. Logs and Sanitizer Results Saved to logs/] ======"
