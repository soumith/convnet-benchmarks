convnet-benchmarks
==================

Easy benchmarking of all public open-source implementations of convnets.
A summary is provided in the section below.

Machine: `6-core Intel i7-3930K @ 3.20GHz` + `NVIDIA Titan Black` + `Ubuntu 14.04 x86_64`

##Imagenet Winners Benchmarking
I pick some popular imagenet models, and I clock the time for a full forward + backward pass. I average my times over 10 runs. I ignored dropout and softmax layers.

AlexNet (One Weird Trick paper) - Input 128x3x224x224
| Original Library         | Class/Function Benchmarked                                                                                               | Total Time (ms)   | Total forward time (ms) | Total backward time (ms) |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                     | **275**           |  **88**                 | **187**                  |
| NVidia CuDNN (R2) *      |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                     | 362               |  100                    | 261                      |
| NVidia CuDNN (R1) *      |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                     | 489               | 134                     | 355                      |
| Torch-7                  | [SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                | 479               |  172                    | 306                      |


Overfeat [fast] - Input 128x3x231x231
| Original Library         | Class/Function Benchmarked                                                                                               | Total Time (ms)   | Total forward time (ms) | Total backward time (ms) |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                     | **1191**          |  385                    | **805**                  |
| NVidia CuDNN (R2) *      |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                     | 1258              |  **308**                | 949                      |
| NVidia CuDNN (R1) *      |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                     | 1480              | 411                     | 1068                     |
| Torch-7                  | [SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                | 1489              |  629                    | 860                      |


OxfordNet [Model-A] - Input 32x3x224x224
| Original Library         | Class/Function Benchmarked                                                                                               | Total Time (ms)   | Total forward time (ms) | Total backward time (ms) |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                     | 1671              | 681                     | 990                      |
| NVidia CuDNN (R2) *      |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                     | **802**           |  **221**                | 581                      |
| NVidia CuDNN (R1) *      |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                     | 952               | 288                     | 663                      |
| Torch-7                  | [SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                | 841               |  262                    | **579**                  |


## Layer-wise Benchmarking

###Spatial Convolution layer (3D input 3D output, densely connected)
##### forward + backprop (wrt input and weights)

| Original Library         | Class/Function Benchmarked                                                                                               | Total Time (ms)   | Total forward time (ms) | Total backward time (ms) | Peak Memory Formula | Limitations |
|:------------------------:|:------------------------------------------------------------------------------------------------------------------------:| -----------------:| -----------------------:| ------------------------:| -------------------:| :---------: |
| Theano (experimental)*** | [conv2d_fft](https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/fftconv.py)                                | **1178**          |  **304**                | **874**                  |                     |             |
| NVidia CuDNN (R2) *      |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                     | 1512              |  413                    | 1099                     |                     |             |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                     | 1551              |  445                    | 1106                     |                     |             |
| Caffe                    | [ConvolutionLayer](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)                             | 1787              |  537                    |   1250                   |                     |             |
| NVidia CuDNN (R1) *      |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                     | 1861              | 513                     | 1348                     |                     |             |
| Torch-7                  |[nn.SpatialConvolutionBHWD](https://github.com/qassemoquab/nnbhwd/blob/master/SpatialConvolutionBHWD.lua)                 | 1892              |  581                    | 1311                     |                     |             |
| Torch-7                  | [SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                | 1936              |  581                    | 1355                     |                     |             |
| Theano (experimental)    | CorrMM                                                                                                                   | 2063              |  630                    | 1433                     |                     |             |
| cuda-convnet**           | [pylearn2.cuda_convnet](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu)   | 3287              |  727                    | 2560                     |                     |             |
| ccv                      | [ccv_convnet_layer](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu)                                 | 809+bw            |  809                    |                          |                     |             |
| Theano (legacy)          | [conv2d](https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/blas.py#L674)                                  | 70774             |  3833                   | 66941                    |                     |             |
| _cherry-picking_****     | _best per layer_                                                                                                         | _985_             |  _191_                  |   _794_                  |                     |             |

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
| Theano (experimental)*** | [conv2d_fft](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.sandbox.cuda.fftconv.conv2d_fft)        | 138 | 73   |  30 | 9  |  39 |   304 |
| NVidia CuDNN (R2)        |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                              | 90  | 218  | 79  | 9  | 17  | 413   |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                              | 63  | 252  |  98 | 11 | 21  |   445 |
| Caffe                    | [ConvolutionLayer\<Dtype>](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)                              | 100 | 205  | 158 | 35 | 39  |  537  |
| NVidia CuDNN (R1)        |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                              | 94  | 274  | 101 | 12 | 32  |   513 |
| Torch-7                  |[nn.SpatialConvolutionBHWD](https://github.com/qassemoquab/nnbhwd/blob/master/SpatialConvolutionBHWD.lua)                          | 182 | 279  | 94  | 11 | 15  |   581 |
| Torch-7                  |[nn.SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                       | 105 | 239  | 168 | 32 | 37  |   581 |
| Theano (experimental)    | CorrMM                                                                                                                            | 100 | 251  | 197 | 38 |  44 |   630 |
| cuda-convnet**           | [pylearn2.cuda_convnet](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu)            | 92  | 412  | 159 | 19 | 45  |   727 |
| ccv                      |[ccv_convnet_layer](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu)                                           | 121 | 437  | 182 | 23 | 44  |   809 |
| Theano (legacy)          | [conv2d](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d)                    | 408 | 2310 | 739 | 99 | 277 |  3833 |
| _cherry-picking_****     | _best per layer_                                                                                                                  | _63_|_72_  | _30_|_9_ | _17_|  191  |

###### backward (gradInput + gradWeight)
Columns L1, L2, L3, L4, L5, Total are times in **milliseconds**

| Original Library         | Class/Function Benchmarked                                                                                                        |  L1   |   L2 |  L3 | L4  |  L5  | Total |
|:------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:| -----:| ----:| ---:| ---:| ----:| -----:|
| Theano (experimental)*** | [conv2d_fft](https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/fftconv.py)                                         | 449   |  218 | 89  | 28  |  90  | 874   |
| NVidia CuDNN (R2)        |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                              | 189   | 606  | 230 | 27  | 47   | 1099  |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                              | 224   | 620  | 200 | 20  | 42   | 1106  |
| Caffe                    | [ConvolutionLayer\<Dtype>](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)                              | 307   |  599 | 242 | 42  |  60  | 1250  |
| NVidia CuDNN (R1)        |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                              | 226   | 736  | 297 | 32  | 57   | 1348  |
| Torch-7                  |[nn.SpatialConvolutionBHWD](https://github.com/qassemoquab/nnbhwd/blob/master/SpatialConvolutionBHWD.lua)                          |  513  | 562  | 187 | 21  | 28   | 1311  | 
| Torch-7                  |[nn.SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                       |  301  |  673 | 270 | 47  | 64   | 1355  |
| Theano (experimental)    | CorrMM                                                                                                                            | 282   | 733  | 295 | 51  |  72  | 1433  |
| cuda-convnet**           | [pylearn2.cuda_convnet](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu)            | 618   | 1305 | 473 | 50  | 114  | 2560  |
| ccv                      |[ccv_convnet_layer](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu)                                           |
| Theano (legacy)          | [conv2d](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d)                    | 53997 | 9752 | 2202 | 299| 691 | 66941  |
| _cherry-picking_****     | _best per layer_                                                                                                                  | _285_ | _337_| _118_|_17_| _37_| _794_  |

###### gradInput
Columns L1, L2, L3, L4, L5, Total are times in **milliseconds**

| Original Library         | Class/Function Benchmarked                                                                                                        |  L1   |   L2 |  L3 | L4  |  L5 | Total |
|:------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:| -----:| ----:| ---:| ---:| ---:| -----:|
| Theano (experimental)*** | [conv2d_fft](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.sandbox.cuda.fftconv.conv2d_fft)        | 250   |  111 |  54 | 19  | 48  |   482 |
| NVidia CuDNN (R2)        |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                              | 91    | 344  | 130 | 15  | 20  |   600 |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                              | 132   |  247 | 90  | 9   | 19  |   497 |
| Caffe                    | [ConvolutionLayer\<Dtype>](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)                              | 86    |  271 | 120 | 20  | 26  |   523 |
| Theano (experimental)    | CorrMM                                                                                                                            | 87    | 328  | 142 | 25  | 31  |   613 |
| NVidia CuDNN (R1)        |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                              | 111   | 421  | 180 | 17  | 21  |   750 |
| Torch-7                  |[nn.SpatialConvolutionBHWD](https://github.com/qassemoquab/nnbhwd/blob/master/SpatialConvolutionBHWD.lua)                          | 276   | 277  | 102  | 11 | 14  |  680  |
| Torch-7                  |[nn.SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                       | 91    |  302 | 129 | 23  | 27  |   572 |
| cuda-convnet**           | [pylearn2.cuda_convnet](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu)            | 155   |  647 | 230 | 23  | 47  |  1102 |
| ccv                      |[ccv_convnet_layer](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu)                                           |
| Theano (legacy)          | [conv2d](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d)                    | 53340 | 2690 | 1044 | 171| 406 | 57651 |
| _cherry-picking_****     | _best per layer_                                                                                                                  | _86_  | _230_| _82_ |_8_ | _16_| _422_ |

###### gradWeights
Columns L1, L2, L3, L4, L5, Total are times in **milliseconds**

| Original Library         | Class/Function Benchmarked                                                                                                        |  L1 |   L2 |  L3  | L4  |  L5 | Total |
|:------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:| ---:| ----:| ----:| ---:| ---:| -----:|
| Theano (experimental)*** | [conv2d_fft](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.sandbox.cuda.fftconv.conv2d_fft)        | 199 | 107  | 35   | 9   | 42  |   392 |
| NVidia CuDNN (R2)        |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                              | 98  | 262  | 100  | 12  | 27  |   499 |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                              | 92  | 373  | 110  | 11  | 23  |  609  |
| Caffe                    | [ConvolutionLayer\<Dtype>](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)                              | 221 | 328  | 122  | 22  | 34  |   727 |
| NVidia CuDNN (R1)        |[cudnn.SpatialConvolution](https://github.com/soumith/cudnn.torch/blob/master/SpatialConvolution.lua)                              | 115 | 315  | 117  | 15  | 36  | 598   |
| Torch-7                  |[nn.SpatialConvolutionBHWD](https://github.com/qassemoquab/nnbhwd/blob/master/SpatialConvolutionBHWD.lua)                          | 237 | 285  | 85   | 10  | 14  |   631 |
| Torch-7                  | [nn.SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                      | 210 | 371  | 141  | 24  | 37  |   783 |
| Theano (experimental)    | CorrMM                                                                                                                            | 195 | 405  | 153  | 26  | 41  |  820  |
| cuda-convnet**           | [pylearn2.cuda_convnet](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu)            | 463 | 658  | 243  | 27  | 67  |  2069 |
| ccv                      | [ccv_convnet_layer](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu)                                          |
| Theano (legacy)          | [conv2d](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d)                    | 657 | 7062 | 1158 | 128 | 285 |  9290 |
| _cherry-picking_****     | _best per layer_                                                                                                                  |_199_| _107_| _36_ | _9_ | _21_|   372 |


