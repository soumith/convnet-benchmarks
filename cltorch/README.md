Install cltorch using the script:
```bash
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash
luarocks install cltorch # to install cltorch matrix layer
luarocks install clnn    # to install clnn network layer
```

For imagenet_winners benchmarks (alexnet, overfeat, vgg and googlenet)
Run the benchmark using:
```bash
th imagenet_winners/benchmark.lua
```

For layerwise benchmarks (table in frontpage with L1,L2,L3,L4,L5)
Run the benchmark using:
```bash
git clone --recursive https://github.com/hughperkins/clnn
cd clnn
th test/test-perf.lua
```

