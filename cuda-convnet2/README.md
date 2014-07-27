After a brief email exchange with Alex, he suggested that the easiest way 
to do benchmarking is to write a small C/C++ wrapper around cudaconv3 (where all the kernels are).
I took this route, except that I wrote a Torch wrapper around the kernels, the repository can be found at 
https://github.com/soumith/cuda-convnet2.torch

Assuming torch is already installed, it can be installed with
luarocks install https://raw.githubusercontent.com/soumith/cuda-convnet2.torch/master/ccn2-scm-1.rockspec

The benchmark will be added to [benchmark.lua in the torch7 folder](https://github.com/soumith/convnet-benchmarks/tree/master/torch7)