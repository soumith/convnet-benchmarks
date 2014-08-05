Install Caffe using the script:
```bash
bash install.sh
```

Run the benchmark using:
```bash
./caffe/build/tools/caffe.bin speedtest --net_proto_file=conv.prototxt --run_iterations=1 --speedtest_with_gpu --logtostderr=1
./caffe/build/tools/caffe.bin speedtest --net_proto_file=conv2.prototxt --run_iterations=1 --speedtest_with_gpu --logtostderr=1
./caffe/build/tools/caffe.bin speedtest --net_proto_file=conv3.prototxt --run_iterations=1 --speedtest_with_gpu --logtostderr=1
./caffe/build/tools/caffe.bin speedtest --net_proto_file=conv4.prototxt --run_iterations=1 --speedtest_with_gpu --logtostderr=1
./caffe/build/tools/caffe.bin speedtest --net_proto_file=conv5.prototxt --run_iterations=1 --speedtest_with_gpu --logtostderr=1
```
