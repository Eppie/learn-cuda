NVCC      := nvcc
ARCH      := -arch=sm_89
NVCCFLAGS := -O3 -std=c++17 $(ARCH) -lineinfo -I../common

BINS := bench starter persistent_demo events_demo ring_buffer

.PHONY: all clean
all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
