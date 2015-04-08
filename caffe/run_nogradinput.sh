#!/bin/bash

./caffe/build/tools/caffe time --model=proto_noGradInput/conv1.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_noGradInput.log 2>&1
./caffe/build/tools/caffe time --model=proto_noGradInput/conv2.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_noGradInput.log 2>&1
./caffe/build/tools/caffe time --model=proto_noGradInput/conv3.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_noGradInput.log 2>&1
./caffe/build/tools/caffe time --model=proto_noGradInput/conv4.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_noGradInput.log 2>&1
./caffe/build/tools/caffe time --model=proto_noGradInput/conv5.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_noGradInput.log 2>&1

