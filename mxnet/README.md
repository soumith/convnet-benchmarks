### Install via:


```
sudo apt-get update
sudo apt-get install -y build-essential git libatlas-base-dev libopencv-dev

git clone --recursive https://github.com/dmlc/mxnet

make -j12 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda
cd python; python setup.py install
cd ../../
```

### Run benchmarks

```
CUDA_VISIBLE_DEVICES=2 MXNET_GPU_WORKER_NTHREADS=2 MXNET_EXEC_NUM_TEMP=1 python alexnet.py | tee out_alexnet.log

```

### Notes from antinucleon

We choose to block the dynamic thread engine to get fair result, definitely there will be some costs,
and I do think it is not worth for these microseconds but the most important thing is learn spirit
from each tools.
The first epoch will make some delay for lazy allocation, but we think it is not a problem.
Also there is a magic number of 4GB threshold for dynamic memory recycle,
we didn't change it although dynamic memory recycle will hurt performance too much.


One import thing about MXNet is chose parallelism level.
Basically, less parallelism, fewer memory cost.
For example, on Titan X with 12 GB memory, train on GoogLeNet v1,

```MXNET_GPU_WORKER_NTHREADS=2 MXNET_EXEC_NUM_TEMP=1 python3 gnet.py``` allows training in batch of 256,

but

```MXNET_GPU_WORKER_NTHREADS=4 MXNET_EXEC_NUM_TEMP=4 python3 gnet.py```

will be oom for batch of 256 (guess still saving a little more than other library but not tested)


Various of setting can be found at: https://mxnet.readthedocs.org/en/latest/env_var.html

In my feeling because currently hardware is bottleneck, dynamic data flow engine and
multi-execution doesn't show its advantage on single card, but in multi-gpu
and distributed case, it makes problem much easier.


BTW. Do you have plan to benchmark multi-gpu or distributed convolution net?
We have collected some result already.

https://github.com/dmlc/mxnet/tree/master/example/distributed-training


