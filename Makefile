CXX = g++
CXXFLAGS = -O2 -std=c++11 -Wall

all: bin/expint_exec

bin/expint_exec: main.cpp expint_cpu.hpp | bin
	$(CXX) $(CXXFLAGS) main.cpp -o bin/expint_exec

bin:
	mkdir -p bin

clean:
	rm -rf bin
