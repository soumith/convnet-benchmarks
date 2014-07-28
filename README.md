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

Since this repository is getting a little attention, quickly adding some more results without making them pretty:
```
CONFIG: input = 3x128x128 * ker = 3x96x11x11 (bs = 128, stride = 1)
cuda-convnet: 942.8129891143 GFLOP/s (tm = 0.13175171613693)
Caffe/nn.SpatialConvolutionMM: 1179.3990140955 GFLOP/s (tm = 0.10532248020172)
cuda-convnet2: 1654.2947261421 GFLOP/s (tm = 0.07508772611618)

CONFIG: input = 64x64x64 * ker = 64x128x9x9 (bs = 128, stride = 1)
cuda-convnet: 1260.0636701749 GFLOP/s (tm = 0.42276448011398)
Caffe/nn.SpatialConvolutionMM: 2221.9725336101 GFLOP/s (tm = 0.23974651098251)
cuda-convnet2: 2202.7792345295 GFLOP/s (tm = 0.24183547496796)

CONFIG: input = 128x32x32 * ker = 128x128x9x9 (bs = 128, stride = 1)
cuda-convnet: 1263.7233233216 GFLOP/s (tm = 0.15485149621964)
Caffe/nn.SpatialConvolutionMM: 1170.9672895372 GFLOP/s (tm = 0.16711777448654)
cuda-convnet2: 2231.4079892349 GFLOP/s (tm = 0.087697744369507)

CONFIG: input = 128x16x16 * ker = 128x128x7x7 (bs = 128, stride = 1)
cuda-convnet: 1193.9667318945 GFLOP/s (tm = 0.01721328496933)
Caffe/nn.SpatialConvolutionMM: 509.81966887597 GFLOP/s (tm = 0.040312469005585)
cuda-convnet2: 2141.0058210269 GFLOP/s (tm = 0.0095992684364319)

CONFIG: input = 384x13x13 * ker = 384x384x3x3 (bs = 128, stride = 1)
cuda-convnet: 976.73603963865 GFLOP/s (tm = 0.042087495326996)
Caffe/nn.SpatialConvolutionMM: 739.60915914596 GFLOP/s (tm = 0.055581212043762)
cuda-convnet2: 2289.1113239623 GFLOP/s (tm = 0.017958223819733)
```
