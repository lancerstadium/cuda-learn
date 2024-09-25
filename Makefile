

NVCC		:= nvcc
SRC			:= demo01

all: $(SRC).out

$(SRC).out: $(SRC).cu
	$(NVCC) $< -o $@

clean:
	rm -rf *.out