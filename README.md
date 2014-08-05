convnet-benchmarks
==================

Easy benchmarking of all public open-source implementations of convnets.
A summary is provided in the section below.


##### Work in progress! I am still working through each convolution module in each library, THIS IS NOT AN EXHAUSTIVE LIST!

* After getting an initial baseline with the single module below (and getting inital benchmark scripts), I will benchmark a full AlexNet/MattNet/Overfeat 

Machine: `6-core Intel i7-3930K @ 3.20GHz` + `NVIDIA Titan Black` + `Ubuntu 14.04 x86_64`

###Spatial Convolution layer (3D input 3D output, densely connected)
##### forward + backprop (wrt input and weights)

| Original Library         | Class/Function Benchmarked                                                                                               | Total Time (ms)   | Total forward time (ms) | Total backward time (ms) | Peak Memory Formula | Limitations |
|:------------------------:|:------------------------------------------------------------------------------------------------------------------------:| -----------------:| -----------------------:| ------------------------:| -------------------:| :---------: |
| Caffe                    | [ConvolutionLayer\<Dtype>](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)                     | **1253**          |  554                    | **699**                  |                     |             |
| Theano (experimental)*** | [conv2d_fft](https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/fftconv.py)                                | 1819              |  **326**                | 1493                     |                     |             |
| Torch-7                  | [nn.SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                             | 2096              |  609                    | 1487                     |                     |             |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                     | 2174              |  424                    | 1750                     |                     |             |
| cuda-convnet**           | [pylearn2.cuda_convnet](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu)   | 3287              |  727                    | 2560                     |                     |             |
| ccv                      | [ccv_convnet_layer](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu)                                 | 809+bw            |  809                    |                          |                     |             |
| Theano (legacy)          | [conv2d](https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/blas.py#L674)                                  | 70774             |  3833                   | 66941                    |                     |             |
| _cherry-picking_****     | _best per layer_                                                                                                         | _870_             |  _192_                  |   _678_                  |                     |             |

* \* indicates that the library was tested with Torch bindings of the specific kernels.
* ** indicates that the library was tested with Pylearn2 bindings. 
* *** This is an experimental module which used FFT to calculate convolutions. [It uses a lot of memory according to @benanne](https://github.com/soumith/convnet-benchmarks/pull/5#issuecomment-50548946)
* **** The last row shows results obtainable when choosing the best-performing library for each layer.
* L1 - Input: `128x128` Batch-size `128`, Feature maps:    `3->96`,  Kernel Size: `11x11`,  Stride: `1x1`
* L2 - Input: `64x64`   Batch-size `128`, Feature maps:  `64->128`,  Kernel Size:   `9x9`,  Stride: `1x1`
* L3 - Input: `32x32`   Batch-size `128`, Feature maps: `128->128`,  Kernel Size:   `9x9`,  Stride: `1x1`
* L4 - Input: `16x16`   Batch-size `128`, Feature maps: `128->128`,  Kernel Size:   `7x7`,  Stride: `1x1`
* L5 - Input: `13x13`   Batch-size `128`, Feature maps: `384->384`,  Kernel Size:   `3x3`,  Stride: `1x1`
* The table is ranked according to the total time forward+backward calls for layers (L1 + L2 + L3 + L4 + L5)

#####Breakdown
###### forward
Columns L1, L2, L3, L4, L5, Total are times in **milliseconds**

| Original Library         | Class/Function Benchmarked                                                                                                        |  L1 |   L2 |  L3 | L4 |  L5 | Total |
|:------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:| ---:| ----:| ---:| --:| ---:| -----:|
| Caffe                    | [ConvolutionLayer\<Dtype>](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)                              | 102 | 203  | 158 | 39 | 52  |   554 |
| Theano (experimental)*** | [conv2d_fft](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.sandbox.cuda.fftconv.conv2d_fft)        | 204 | 76   |  31 | 10 |  5  |   326 |
| Torch-7                  |[nn.SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                       | 105 | 242  | 168 | 50 | 56  |   609 |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                              | 70  | 244  |  87 | 11 | 18  |   424 |
| cuda-convnet**           | [pylearn2.cuda_convnet](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu)            | 92  | 412  | 159 | 19 | 45  |   727 |
| ccv                      |[ccv_convnet_layer](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu)                                           | 121 | 437  | 182 | 23 | 44  |   809 |
| Theano (legacy)          | [conv2d](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d)                    | 408 | 2310 | 739 | 99 | 277 |  3833 |
| _cherry-picking_****     | _best per layer_                                                                                                                  | _70_|_76_  | _31_|_10_|  _5_|  192  |

###### backward (gradInput + gradWeight)
Columns L1, L2, L3, L4, L5, Total are times in **milliseconds**

| Original Library         | Class/Function Benchmarked                                                                                                        |  L1   |   L2 |  L3 | L4  |  L5  | Total |
|:------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:| -----:| ----:| ---:| ---:| ----:| -----:|
| Caffe                    | [ConvolutionLayer\<Dtype>](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)                              | 194   |  328 | 122 | 21  | 34   |  699  |
| Theano (experimental)*** | [pylearn2.mlp.ConvElemwise](https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/fftconv.py)                          | 931   |  370 | 137 | 42  |  13  | 1493  |
| Torch-7                  |[nn.SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                       | 380   |  682 | 293 | 55  | 77   | 1487  |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                              | 797   |  645 | 238 | 26  | 44   | 1750  |
| cuda-convnet**           | [pylearn2.cuda_convnet](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu)            | 618   | 1305 | 473 | 50  | 114  | 2560  |
| ccv                      |[ccv_convnet_layer](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu)                                           |
| Theano (legacy)          | [conv2d](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d)                    | 53997 | 9752 | 2202 | 299| 691 | 66941 |
| _cherry-picking_****     | _best per layer_                                                                                                                  | _194_ | _328_| _122_|_21_| _13_| _678_ |

###### gradInput
Columns L1, L2, L3, L4, L5, Total are times in **milliseconds**

| Original Library         | Class/Function Benchmarked                                                                                                        |  L1   |   L2 |  L3 | L4  |  L5 | Total |
|:------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:| -----:| ----:| ---:| ---:| ---:| -----:|
| Caffe                    | [ConvolutionLayer\<Dtype>](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)                              |
| Theano (experimental)*** | [conv2d_fft](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.sandbox.cuda.fftconv.conv2d_fft)        | 730   |  258 | 101 | 32  |  7  |  1128 |
| Torch-7                  |[nn.SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                       | 91    |  307 | 133 | 27  | 27  |   585 |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                              | 155   |  300 | 118 | 13  | 22  |   608 |
| cuda-convnet**           | [pylearn2.cuda_convnet](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu)            | 155   |  647 | 230 | 23  | 47  |  1102 |
| ccv                      |[ccv_convnet_layer](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu)                                           |
| Theano (legacy)          | [conv2d](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d)                    | 53340 | 2690 | 1044 | 171| 406 | 57651 |
| _cherry-picking_****     | _best per layer_                                                                                                                  | _91_  | _258_| _101_|_13_| _7_ | _470_ |

###### gradWeights
Columns L1, L2, L3, L4, L5, Total are times in **milliseconds**

| Original Library         | Class/Function Benchmarked                                                                                                        |  L1 |   L2 |  L3  | L4  |  L5 | Total |
|:------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:| ---:| ----:| ----:| ---:| ---:| -----:|
| Caffe                    | [ConvolutionLayer\<Dtype>](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)                              |
| Theano (experimental)*** | [conv2d_fft](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.sandbox.cuda.fftconv.conv2d_fft)        | 201 | 112  |   36 | 10  | 6   |   365 |
| Torch-7                  | [nn.SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                      | 189 | 375  | 160  | 28  | 50  |   802 |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                              | 642 | 345  | 120  | 13  | 22  |  1142 |
| cuda-convnet**           | [pylearn2.cuda_convnet](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu)            | 463 | 658  | 243  | 27  | 67  |  2069 |
| ccv                      | [ccv_convnet_layer](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu)                                          |
| Theano (legacy)          | [conv2d](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d)                    | 657 | 7062 | 1158 | 128 | 285 |  9290 |
| _cherry-picking_****     | _best per layer_                                                                                                                  |_201_| _112_| _36_ | _10_|_6_  |   365 |


