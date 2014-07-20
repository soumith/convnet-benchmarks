Install
-------
sudo apt-get install python-dev python-numpy python-scipy python-magic python-matplotlib libatlas-base-dev libjpeg-dev libopencv-dev git
git clone https://code.google.com/p/cuda-convnet2/
cd cuda-convnet2
sh build.sh


Run Benchmark
-------------
python convnet.py --layer-def=../bench1.cfg --layer-params=../bench1-params.cfg --data-provider=dummy-labeled-49152 --check-grads=1 --gpu 1
