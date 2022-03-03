.PHONY: all clean check

CUDA_BASE=/usr/local/cuda-11.5
CUDA_ARCH=75


all: test

clean:
	rm -f test

test: test.cu
	$(CUDA_BASE)/bin/nvcc $< -o $@ -std=c++17 -O3 -g --generate-line-info --generate-code arch=compute_$(CUDA_ARCH),code=[compute_$(CUDA_ARCH),sm_$(CUDA_ARCH)]

check: test
	$(CUDA_BASE)/bin/compute-sanitizer --launch-timeout 0 --kill yes --error-exitcode 127 --require-cuda-init yes --nvtx yes --print-level info --demangle full --report-api-errors all --tool memcheck --leak-check full $<
