Install Caffe using the script:
```bash
bash install.sh
```

Run the benchmark using:
```bash
./caffe/build/tools/caffe time --model=conv1.prototxt --iterations=1 --gpu 0 --logtostderr=1
./caffe/build/tools/caffe time --model=conv2.prototxt --iterations=1 --gpu 0 --logtostderr=1
./caffe/build/tools/caffe time --model=conv3.prototxt --iterations=1 --gpu 0 --logtostderr=1
./caffe/build/tools/caffe time --model=conv4.prototxt --iterations=1 --gpu 0 --logtostderr=1
./caffe/build/tools/caffe time --model=conv5.prototxt --iterations=1 --gpu 0 --logtostderr=1
```
