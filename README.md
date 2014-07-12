convnet-benchmarks
==================

Easy benchmarking of all public open-source implementations of convnets.
A summary is provided in the section below.

Machine: `6-core Intel i7-3930K @ 3.20GHz` + `NVIDIA Titan Black`

###Spatial Convolution layer (3D input 3D output)


#####Configuration: Input: `128x128` Batch-size `128`, Feature maps: `3->96`,  Kernel Size `11x11`
| Library       | Class/Function         | Device | Input Config| GFlop/s   | Code URL       |
|:-------------:|:----------------------:|:-----------:|:-----------:|:---------:|:--------------:|
| Torch-7       | nn.SpatialConvolutionCUDA |GPU| DHWB | 844.72  | [Link](https://github.com/torch/cunn/blob/master/SpatialConvolutionCUDA/updateOutput.cu) |


