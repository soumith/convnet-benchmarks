Install torch-7 using the instructions at: http://torch.ch/docs/getting-started.html#_
Then install cudaconv2, cudnn
```bash
luarocks install ccn2 # to do cuda-convnet2 benchmarks using torch wrappers
luarocks install cudnn # to do NVIDIA CuDNN benchmarks, also have CuDNN installed on your machine
#luarocks install https://raw.githubusercontent.com/qassemoquab/nnbhwd/master/nnbhwd-scm-1.rockspec # to do nnBHWD benchmarks
```

Also install fbcunn's sync branch (I just synced from internal, pending cleanup): https://github.com/facebook/fbcunn/blob/master/INSTALL.md

For layerwise benchmarks (table in frontpage with L1,L2,L3,L4,L5)
Run the benchmark using:
```bash
th layerwise_benchmarks/benchmark.lua
```

For imagenet-winners benchmarks
Run the benchmark using:
```bash
th imagenet_winners/benchmark.lua
```

For imagenet-winners benchmarks of cudaconv2, look at the cudaconv2 branch: https://github.com/soumith/convnet-benchmarks/tree/cudaconv2
