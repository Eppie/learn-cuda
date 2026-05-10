MODULES := 01-execution-model 02-memory-coalescing 03-shared-memory 04-profiling 05-reductions 06-gemm 07-tensor-cores 08-async-copy 09-fused-epilogues 10-flash-attention 11-low-latency 12-capstone 13-ptx-appendix

.PHONY: all clean $(MODULES)

all: $(MODULES)

$(MODULES):
	$(MAKE) -C $@

clean:
	@for d in $(MODULES); do $(MAKE) -C $$d clean; done
