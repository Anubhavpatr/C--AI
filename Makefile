CXX = g++
CXXFLAGS = -std=c++20 -O2

# Adjust this path to your downloaded LibTorch directory optional just external libraries

SRC = main.cpp Logger.cpp Tensor.cpp
OUT = main

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ 

clean:
	rm -f $(OUT)