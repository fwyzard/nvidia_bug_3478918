# nvidia_bug_3478918
A simple program to reproduce NVIDIA bug 3478918

## building and running
```bash
git clone https://github.com/fwyzard/nvidia_bug_3478918.git
cd nvidia_bug_3478918
make
make check
```

## rationale

Calling `cg::this_grid().sync()` in a kernel launched via `cudaLaunchCooperativeKernel()`, while using
multiple host threads and CUDA streams, causes `compute-sanitize` to report an out-of-bounds access in
`cooperative_groups::__v1::details::atomic_add()`:
```
========= COMPUTE-SANITIZER
========= Invalid __global__ atomic of size 4 bytes
=========     at 0x270 in /usr/local/cuda-11.5/targets/x86_64-linux/include/cooperative_groups/details/sync.h:81:_ZN37_INTERNAL_60388108_7_test_cu_3512655418cooperative_groups4__v17details10atomic_addEPVjj
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7fcccea00204 is out of bounds
=========     and is 5 bytes after the nearest allocation at 0x7fcccea00000 of size 512 bytes
=========     Device Frame:/usr/local/cuda-11.5/targets/x86_64-linux/include/cooperative_groups/details/sync.h:101:_ZN37_INTERNAL_60388108_7_test_cu_3512655418cooperative_groups4__v17details10sync_gridsEjPVj [0x130]
=========     Device Frame:/usr/local/cuda-11.5/targets/x86_64-linux/include/cooperative_groups/details/helpers.h:336:_ZN37_INTERNAL_60388108_7_test_cu_3512655418cooperative_groups4__v17details4grid4syncEPj [0x50]
=========     Device Frame:/usr/local/cuda-11.5/targets/x86_64-linux/include/cooperative_groups.h:331:cooperative_groups::__v1::grid_group::sync() const [0x50]
=========     Device Frame:/nfshome0/fwyzard/test/test_cg/test.cu:14:kernel(int) [0x10]
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x20828a]
=========                in /lib64/libcuda.so.1
=========     Host Frame:__cudart1329 [0xc955]
=========                in /cmsnfshome0/nfshome0/fwyzard/test/test_cg/test
=========     Host Frame:cudaLaunchCooperativeKernel [0x666c8]
=========                in /cmsnfshome0/nfshome0/fwyzard/test/test_cg/test
=========     ...
```

In our tests, this happens only when using 8 or more concurrent host threads and CUDA streams.
Running with 1-7 host threads and CUDA streams does not raise any errors.

This was tested on a Tesla T4 and an A10, using CUDA 11.2, 11.4 and 11.5. Other GPUs or CUDA versions are
likely affected.
