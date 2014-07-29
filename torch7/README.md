Install torch-7 using the script:
```bash
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash
luarocks install ccn2 # to do cuda-convnet2 benchmarks using torch wrappers
```

Run the benchmark using:
```bash
th benchmark.lua
```
