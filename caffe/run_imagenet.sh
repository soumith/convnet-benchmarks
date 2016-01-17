#!/bin/bash

NUM_THREADS=`cat /proc/cpuinfo | grep 'processor' | wc -l`
echo "Running with ${NUM_THREADS} threads"

export OPENBLAS_NUM_THREADS=${NUM_THREADS}
export MKL_NUM_THREADS=${NUM_THREADS}
export MKL_BLAS_NUM_THREADS=${NUM_THREADS}
export OMP_NUM_THREADS=${NUM_THREADS}


for BLAS in mkl openblas
do
	./caffe/build-with-${BLAS}/tools/caffe time --model=./imagenet_winners/alexnet.prototxt --iterations=10 --logtostderr=1 >>output_alexnet_${BLAS}.log 2>&1
	./caffe/build-with-${BLAS}/tools/caffe time --model=./imagenet_winners/overfeat.prototxt --iterations=10 --logtostderr=1 >>output_overfeat_${BLAS}.log 2>&1
	./caffe/build-with-${BLAS}/tools/caffe time --model=./imagenet_winners/vgg_a.prototxt --iterations=10 --logtostderr=1 >>output_vgg_a_${BLAS}.log 2>&1
	./caffe/build-with-${BLAS}/tools/caffe time --model=./imagenet_winners/googlenet.prototxt --iterations=10 --logtostderr=1 >>output_googlenet_${BLAS}.log 2>&1
done

