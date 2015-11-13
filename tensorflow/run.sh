export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib:/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/Downloads/cudnn-6.5-linux-x64-v2:$LD_LIBRARY_PATH
export PATH="$HOME/code/bazel/bazel-bin/src":$PATH
export CUDA_VISIBLE_DEVICES=2

rm ~/code/tensorflow/tensorflow/models/convnetbenchmarks
ln -s ~/code/convnet-benchmarks/tensorflow ~/code/tensorflow/tensorflow/models/convnetbenchmarks

cd ~/code/tensorflow

bazel build -c opt --config=cuda //tensorflow/models/convnetbenchmarks:benchmark_alexnet
bazel build -c opt --config=cuda //tensorflow/models/convnetbenchmarks:benchmark_overfeat
bazel build -c opt --config=cuda //tensorflow/models/convnetbenchmarks:benchmark_vgg
bazel build -c opt --config=cuda //tensorflow/models/convnetbenchmarks:benchmark_googlenet

bazel-bin/tensorflow/models/convnetbenchmarks/benchmark_alexnet   2>&1 | tee ~/code/convnet-benchmarks/output_alexnet.log
bazel-bin/tensorflow/models/convnetbenchmarks/benchmark_overfeat  2>&1 | tee ~/code/convnet-benchmarks/output_overfeat.log
bazel-bin/tensorflow/models/convnetbenchmarks/benchmark_vgg       2>&1 | tee ~/code/convnet-benchmarks/output_vgga.log
bazel-bin/tensorflow/models/convnetbenchmarks/benchmark_googlenet 2>&1 | tee ~/code/convnet-benchmarks/output_googlenet.log

cd ~/code/convnet-benchmarks/tensorflow
