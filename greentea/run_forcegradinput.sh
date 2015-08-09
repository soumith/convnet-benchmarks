#!/bin/bash

./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_forceGradInput.log 2>&1
./caffe/build/tools/caffe time --model=proto_forceGradInput/conv2.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_forceGradInput.log 2>&1
./caffe/build/tools/caffe time --model=proto_forceGradInput/conv3.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_forceGradInput.log 2>&1
./caffe/build/tools/caffe time --model=proto_forceGradInput/conv4.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_forceGradInput.log 2>&1
./caffe/build/tools/caffe time --model=proto_forceGradInput/conv5.prototxt --iterations=10 --gpu 0 --logtostderr=1 >>output_forceGradInput.log 2>&1
