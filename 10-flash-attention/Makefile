NVCC      := nvcc
ARCH      := -arch=sm_89
NVCCFLAGS := -O3 -std=c++17 $(ARCH) -lineinfo -I../common

BINS := solution starter bench

.PHONY: all clean
all: $(BINS)

%: %.cu kernels.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

# Roofline reference (cuBLAS hgemm at the bench's exact shape).
roofline: roofline.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcublas

.PHONY: roofline-run
roofline-run: roofline
	./roofline
