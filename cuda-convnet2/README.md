After a brief email exchange with Alex, he suggested that the easiest way to do benchmarking is to write a small C/C++ wrapper around cudaconv3 (where all the kernels are).
I took this route, except that I wrote a Torch/FFI wrapper around the kernels (instead of C/C++), the repository can be found at 
https://github.com/soumith/cuda-convnet2.torch

For details on installing torch, look at the README.md in the [torch7 folder](https://github.com/soumith/convnet-benchmarks/tree/master/torch7)
Assuming torch is already installed, it can be installed with
```bash
luarocks install https://raw.githubusercontent.com/soumith/cuda-convnet2.torch/master/ccn2-scm-1.rockspec
```

The benchmark is included with [benchmark.lua in the torch7 folder](https://github.com/soumith/convnet-benchmarks/tree/master/torch7)

The benchmark can be run with the command:
```bash
th benchmark.lua
```


