CXX = g++
NVCC = nvcc
CXXFLAGS = -O2 -std=c++11 -Wall -Iinclude
NVFLAGS = -O2 -std=c++11 -arch=sm_70 -Iinclude

SRC = main.cpp
CU_SRC = src/expint_gpu.cu

all: bin/expint_exec

bin/expint_exec: $(SRC) $(CU_SRC) include/expint_cpu.hpp include/expint_gpu.hpp
	$(NVCC) $(NVFLAGS) $(SRC) $(CU_SRC) -o bin/expint_exec

clean:
	rm -rf bin/expint_exec
