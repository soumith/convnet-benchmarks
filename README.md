convnet-benchmarks
==================

Easy benchmarking of all public open-source implementations of convnets.
A summary is provided in the section below.


##### Work in progress! I am still working through each convolution module in each library, THIS IS NOT AN EXHAUSTIVE LIST!

* After getting an initial baseline with the single module below (and getting inital benchmark scripts), I will benchmark a full AlexNet/MattNet/Overfeat 

Machine: `6-core Intel i7-3930K @ 3.20GHz` + `NVIDIA Titan Black` + `Ubuntu 14.04 x86_64`

###Spatial Convolution layer (3D input 3D output)
#####:forward()
Columns L1, L2, L3, L4, L5, Total are times in **miliseconds**

| Original Library         | Class/Function Benchmarked                                                                                                        | Device  |  L1 |  L2 |  L3 |  L4 | L5  | Total |
|:------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:-------:|:---:|:----:|:---:|:--:|:---:|:-----:|
| Theano (experimental)*** | [pylearn2.mlp.ConvElemwise](https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/fftconv.py)                          |  GPU    | 205 | 75   |  28 |  9 | 5   |   322 |
| cuda-convnet2 *          | [ConvLayer](https://github.com/soumith/cuda-convnet2.torch/blob/master/cudaconv3/src/filter_acts.cu)                              |  GPU    | 69  | 242  |  87 |  9 | 17  |   424 |
| Caffe                    | [ConvolutionLayer\<Dtype>](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)                              |  GPU    | 102 |  203 | 158 | 39 |  52 |   554 |
| cuda-convnet**           | [pylearn2.cuda_convnet](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/cuda_convnet/filter_acts.cu)  |  GPU    | 98  | 404  | 149 | 16 | 38  |   705 |
| Torch-7                  |[nn.SpatialConvolutionMM](https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu)                                       |  GPU    | 105 | 240  | 168 | 41 | 55  |   609 |
| ccv                      |[ccv_convnet_layer](https://github.com/liuliu/ccv/blob/unstable/lib/cuda/cwc_convnet.cu)                                           |  GPU    | 153 |      |     |    |     |       |
| Theano (legacy)**        | [pylearn2.mlp.ConvElemwise](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/models/mlp.py#L3080)                        |  GPU    | 418 | 2299 | 672 | 88 | 272 |  3749 |


* \* indicates that the library was tested with Torch bindings of the specific kernels.
* ** indicates that the library was tested with Pylearn2 bindings. 
* *** This is an experimental module which used FFT to calculate convolutions. [It uses a lot of memory according to @benanne](https://github.com/soumith/convnet-benchmarks/pull/5#issuecomment-50548946)
* L1 - Input: `128x128` Batch-size `128`, Feature maps:    `3->96`,  Kernel Size: `11x11`,  Stride: `1x1`
* L2 - Input: `64x64`   Batch-size `128`, Feature maps:  `64->128`,  Kernel Size:   `9x9`,  Stride: `1x1`
* L3 - Input: `32x32`   Batch-size `128`, Feature maps: `128->128`,  Kernel Size:   `9x9`,  Stride: `1x1`
* L4 - Input: `16x16`   Batch-size `128`, Feature maps: `128->128`,  Kernel Size:   `7x7`,  Stride: `1x1`
* L5 - Input: `13x13`   Batch-size `128`, Feature maps: `384->384`,  Kernel Size:   `3x3`,  Stride: `1x1`
* The table is ranked according to the total time (L1 + L2 + L3 + L4 + L5)


