#!/bin/bash

./caffe/build/tools/caffe time --model=./imagenet_winners/alexnet.prototxt --iterations=10 --gpu 0
./caffe/build/tools/caffe time --model=./imagenet_winners/overfeat.prototxt --iterations=10 --gpu 0
./caffe/build/tools/caffe time --model=./imagenet_winners/vgg_a.prototxt --iterations=10 --gpu 0
./caffe/build/tools/caffe time --model=./imagenet_winners/googlenet.prototxt --iterations=10 --gpu 0

