convnet-benchmarks
==================

Easy benchmarking of all public open-source implementations of convnets.
A summary is provided in the section below.

* Work in progress! After getting an initial baseline with the single module below (and getting inital benchmark scripts), I will benchmark a full AlexNet/MattNet/Overfeat *

Machine: `6-core Intel i7-3930K @ 3.20GHz` + `NVIDIA Titan Black` + `Ubuntu 14.04 x86_64`

###Spatial Convolution layer (3D input 3D output)


#####Configuration: Input: `128x128` Batch-size `128`, Feature maps: `3->96`,  Kernel Size `11x11`

#####:forward()
| Library         | Class/Function                      | Device | Input Config   | GFlop/s   | Code URL       |
|:-------------:  |:-----------------------------------:|:------:|:--------------:|:---------:|:--------------:|
| cuda-convnet2 * | ConvLayer                           |GPU     | DHWB           | 1642.15 | [Link](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu) |
| Caffe           | ConvolutionLayer\<Dtype>            |GPU     | BDHW           | 1258.70 | [Link](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu) |
| Torch-7         | nn.SpatialConvolutionMM             |GPU     | BDHW           | 1151.24 | [Link](https://github.com/torch/cunn/blob/spatialconvmm/SpatialConvolutionMM.cu) |
| ccv             | ccv_convnet_layer                   |GPU     | BDHW           | 1024.16 | [Link](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu) |
| cuda-convnet *  | ConvLayer                           |GPU     | DHWB           | 929.17  | [Link](https://github.com/torch/cunn/blob/master/SpatialConvolutionCUDA/updateOutput.cu) |
| Theano/pylearn2 | pylearn2.models.mlp.ConvElemwise  |GPU     | BDHW           | 298.14  | [Link](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/models/mlp.py#L3080) |

A * indicates that the library was tested with Torch bindings of the specific kernels.
