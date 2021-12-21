#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <future>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "cuda_check.h"

__global__ void kernel(int n) {
  namespace cg = cooperative_groups;
  auto grid = cg::this_grid();
  grid.sync();
}

template <typename F>
inline int maxCoopBlocks(F kernel, int blockSize, int shmem) {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  int numBlocksPerSm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, blockSize, shmem));

  return numBlocksPerSm * properties.multiProcessorCount;
}

int main() {
  const int elements = 1000;
  const int threads = 8;

  int blockSize = 128;
  int gridSize = (elements + blockSize - 1) / blockSize;
  assert(gridSize > 0);

  int maxCoopBlocksForKernel = maxCoopBlocks(kernel, blockSize, 0);
  gridSize = std::min(gridSize, maxCoopBlocksForKernel);
  assert(gridSize > 0);

  void const* args[] = {&elements};
  dim3 block(blockSize, 1, 1);
  dim3 grid(gridSize, 1, 1);
  std::vector<std::future<void>> futures;
  for (int i = 0; i < threads; ++i) {
    futures.emplace_back(std::async([=]() {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel, grid, block, (void**)args, 0, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }));
  }

  for (int i = 0; i < threads; ++i) {
    futures[i].wait();
  }

  return 0;
}
