NVCC      := nvcc
ARCH      := -arch=sm_89
NVCCFLAGS := -O3 -std=c++17 $(ARCH) -lineinfo -I../common

BINS := solution starter bench scan_solution scan_starter scan_bench

.PHONY: all clean
all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
