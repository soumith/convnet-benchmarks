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
| cuda-convnet2 *    | ConvLayer                           |GPU     | DHWB           | 1779.29 | [Link](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu) |
| Caffe              | ConvolutionLayer\<Dtype>            |GPU     | BDHW           | 1258.70 | [Link](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu) |
| cuda-convnet**     | pylearn2..cuda_convnet/ConvLayer     |GPU     | DHWB           | 1202.65 | [Link](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu) |
| Torch-7            | nn.SpatialConvolutionMM             |GPU     | BDHW           | 1177.78 | [Link](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu) |
| ccv                | ccv_convnet_layer                   |GPU     | BDHW           | 1024.16 | [Link](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu) |
| pylearn2/Theano*** | pylearn2..mlp.ConvElemwise    |GPU     | BDHW           | 299.48  | [Link](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/models/mlp.py#L3080) |


* * indicates that the library was tested with Torch bindings of the specific kernels.
* ** indicates that the library was tested with Pylearn2 bindings. 
* *** Ian Goodfellow from LISA Lab informs me that for pylearn2/Theano, they usually use the cuda-convnet binding (the entry with **), and this kernel was only kept around for non-standard shapes that cuda-convnet does not support

**Since this repository is getting a little attention, quickly adding some more results without making them pretty:
cuda-convnet2 blows the competition out of the water by a huge margin!**
```
-- layer 1
CONFIG: input = 3x128x128 * ker = 3x96x11x11 (bs = 128, stride = 1)
cuda-convnet: 944.68789645199 GFLOP/s (tm = 0.1314902305603)
Caffe/nn.SpatialConvolutionMM: 1177.7879871761 GFLOP/s (tm = 0.10546654462814)
cuda-convnet2: 1779.2908234941 GFLOP/s (tm = 0.069812774658203)

-- layer 2
CONFIG: input = 64x64x64 * ker = 64x128x9x9 (bs = 128, stride = 1)
cuda-convnet: 1260.782333477 GFLOP/s (tm = 0.42252349853516)
Caffe/nn.SpatialConvolutionMM: 2219.3736959591 GFLOP/s (tm = 0.24002724885941)
cuda-convnet2: 2197.8168036727 GFLOP/s (tm = 0.24238151311874)

-- layer 3
CONFIG: input = 128x32x32 * ker = 128x128x9x9 (bs = 128, stride = 1)
cuda-convnet: 1259.195897222 GFLOP/s (tm = 0.15540826320648)
Caffe/nn.SpatialConvolutionMM: 1164.72445476 GFLOP/s (tm = 0.16801351308823)  
cuda-convnet2: 2244.3954637671 GFLOP/s (tm = 0.087190270423889)

-- layer 4
CONFIG: input = 128x16x16 * ker = 128x128x7x7 (bs = 128, stride = 1)
cuda-convnet: 1204.6621963356 GFLOP/s (tm = 0.017060458660126)
Caffe/nn.SpatialConvolutionMM: 490.99599360714 GFLOP/s (tm = 0.041857957839966)
cuda-convnet2: 2108.8329876002 GFLOP/s (tm = 0.009745717048645)

-- layers with small inputs/kernels, seen at the lower ends of the network
CONFIG: input = 384x13x13 * ker = 384x384x3x3 (bs = 128, stride = 1)
cuda-convnet: 983.54876028249 GFLOP/s (tm = 0.041795969009399)
Caffe/nn.SpatialConvolutionMM: 735.82546939461 GFLOP/s (tm = 0.05586701631546)
cuda-convnet2: 2283.9565044269 GFLOP/s (tm = 0.01799875497818)
```
