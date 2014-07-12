convnet-benchmarks
==================

Easy benchmarking of all public open-source implementations of convnets.
A summary is provided in the table below.

Benchmark table for Convolution layers:
Machine: 6-core Intel(R) Core(TM) i7-3930K CPU @ 3.20GHz + NVIDIA Titan Black @ 2880 CUDA cores
GPU Benchmarks (Fastest modules first)

#####Configuration: Input: 128x128 Batch-size 128, Feature maps: 3->96,  Kernel Size 11x11
| Library       | Class/Function         | Input Config| GFlop/s   | Code URL       |
|:-------------:|:----------------------:|:-----------:|:---------:|:--------------:|
| Torch-7       | nn.SpatialConvolutionCUDA | DHWB | 844.72  | [Link](https://github.com/torch/cunn/blob/master/SpatialConvolutionCUDA/updateOutput.cu) |


