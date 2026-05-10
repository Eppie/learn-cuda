NVCC      := nvcc
ARCH      := -arch=sm_89
NVCCFLAGS := -O3 -std=c++17 $(ARCH) -lineinfo -I../common

BINS := solution starter bench starter_mb solution_mb

.PHONY: all clean
all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
