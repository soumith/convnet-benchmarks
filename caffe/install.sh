git clone https://github.com/BVLC/caffe.git
cd caffe
git checkout dev

# Dependencies
sudo apt-get install libatlas-base-dev
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev libgflags-dev

wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
tar zxvf glog-0.3.3.tar.gz
cd glog-0.3.3
./configure
make && sudo make install
cd ..

sudo apt-get install liblmdb-dev

# Compile Caffe
cp Makefile.config.example Makefile.config
# Adjust Makefile.config (for example, if using Anaconda Python)
make all
make test
make runtest
