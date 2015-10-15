export LD_LIBRARY_PATH=/home/awesomebox/code/cvbcpu/protobuf-3.0.0-beta-1/src/.libs:/home/awesomebox/code/cvbcpu/intel_optimized_technical_preview_for_multinode_caffe_1.0/build/lib:/opt/intel/lib/intel64:/home/awesomebox/torch/install/lib:/usr/local/cuda/lib64:/opt/intel/compilers_and_libraries_2016.0.109/linux/mkl/lib/intel64_lin:

./build/tools/caffe time -iterations 10 --model=models/intel_alexnet/alexnet.prototxt --logtostderr=1 >>output_alexnet.log 2>&1
./build/tools/caffe time -iterations 10 --model=models/intel_alexnet/vgg_a.prototxt
