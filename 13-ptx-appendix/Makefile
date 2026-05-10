NVCC      := nvcc
ARCH      := -arch=sm_89
NVCCFLAGS := -O3 -std=c++17 $(ARCH) -lineinfo -I../common

BINS := vector_add cache_hints cpasync_inline mma_sync_example ldmatrix_example clock_microbench bar_sync_example
PTX  := vector_add.ptx cache_hints.ptx cpasync_inline.ptx mma_sync_example.ptx ldmatrix_example.ptx
SASS := vector_add.sass cache_hints.sass cpasync_inline.sass mma_sync_example.sass ldmatrix_example.sass

.PHONY: all clean ptx sass

all: $(BINS)

ptx: $(PTX)

sass: $(SASS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) --ptx -o $@ $<

%.cubin: %.cu
	$(NVCC) $(NVCCFLAGS) --cubin -o $@ $<

%.sass: %.cubin
	cuobjdump --dump-sass $< > $@

clean:
	rm -f $(BINS) *.ptx *.sass *.cubin
