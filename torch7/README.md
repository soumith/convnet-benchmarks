Install torch-7 using the script:
```bash
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash
luarocks install ccn2 # to do cuda-convnet2 benchmarks using torch wrappers
luarocks install cudnn # to do NVIDIA CuDNN benchmarks, also have CuDNN installed on your machine
luarocks install https://raw.githubusercontent.com/qassemoquab/nnbhwd/master/nnbhwd-scm-1.rockspec # to do nnBHWD benchmarks
```

Run the benchmark using:
```bash
th benchmark.lua
```
