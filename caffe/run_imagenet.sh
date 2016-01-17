#!/bin/bash

for BLAS in mkl openblas
do
	./caffe/build-with-${BLAS}/tools/caffe time --model=./imagenet_winners/alexnet.prototxt --iterations=10 --logtostderr=1 >>output_alexnet_${BLAS}.log 2>&1
	./caffe/build-with-${BLAS}/tools/caffe time --model=./imagenet_winners/overfeat.prototxt --iterations=10 --logtostderr=1 >>output_overfeat_${BLAS}.log 2>&1
	./caffe/build-with-${BLAS}/tools/caffe time --model=./imagenet_winners/vgg_a.prototxt --iterations=10 --logtostderr=1 >>output_vgg_a_${BLAS}.log 2>&1
	./caffe/build-with-${BLAS}/tools/caffe time --model=./imagenet_winners/googlenet.prototxt --iterations=10 --logtostderr=1 >>output_googlenet_${BLAS}.log 2>&1
done

