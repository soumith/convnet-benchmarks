convnet-benchmarks
==================

Easy benchmarking of all public open-source implementations of convnets.
A summary is provided in the table below.

Benchmark table for Convolution layers:
Machine: 6-core Intel(R) Core(TM) i7-3930K CPU @ 3.20GHz + NVIDIA Titan Black @ 2880 CUDA cores
GPU Benchmarks (Fastest modules first)
----------------------------------------------------------------------------------------------------------------------
| Library       | Class/Function         | Input               | Feature Maps, Kernel Size | GFlop/s | Code URL       |
|:-------------:|:----------------------:|:---------------------:|:-------------------------:|:---------:|:----------------:|
| Torch-7       | SpatialConvolutionCUDA | 128x128x3x128 (DHWB)| 3 -> 96, 11x11      | 844.72  | [Link](https://github.com/torch/cunn/blob/master/SpatialConvolutionCUDA/updateOutput.cu) |


