git clone https://github.com/BVLC/caffe.git
cd caffe
git checkout dev

# Dependencies
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler
sudo apt-get install gcc-4.6 g++-4.6

export CC=gcc-4.6
export CXX=g++-4.6
# Compile Caffe
cp Makefile.config.example Makefile.config
# Adjust Makefile.config (for example, if using Anaconda Python)
make all
make test
make runtest
