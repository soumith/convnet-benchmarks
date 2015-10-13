#!/bin/bash

./caffe/build/tools/caffe time --model=./imagenet_winners/alexnet.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_alexnet.log 2>&1
./caffe/build/tools/caffe time --model=./imagenet_winners/overfeat.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_overfeat.log 2>&1
./caffe/build/tools/caffe time --model=./imagenet_winners/vgg_a.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_vgg_a.log 2>&1
./caffe/build/tools/caffe time --model=./imagenet_winners/googlenet.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_googlenet.log 2>&1

