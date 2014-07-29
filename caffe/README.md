Install Caffe using the script:
```bash
bash install.sh
```

Run the benchmark using:
```bash
./caffe/build/tools/net_speed_benchmark.bin conv.prototxt 1 GPU
./caffe/build/tools/net_speed_benchmark.bin conv2.prototxt 1 GPU
./caffe/build/tools/net_speed_benchmark.bin conv3.prototxt 1 GPU
./caffe/build/tools/net_speed_benchmark.bin conv4.prototxt 1 GPU
./caffe/build/tools/net_speed_benchmark.bin conv5.prototxt 1 GPU
```
