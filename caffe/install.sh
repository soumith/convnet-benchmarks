git clone https://github.com/BVLC/caffe.git
cd caffe

# Dependencies
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev 
sudo apt-get install -y protobuf-compiler gfortran libjpeg62 libfreeimage-dev libatlas-base-dev git python-dev python-pip 
sudo apt-get install -y libgoogle-glog-dev libbz2-dev libxml2-dev libxslt-dev libffi-dev libssl-dev libgflags-dev liblmdb-dev python-yaml
sudo easy_install pillow

# Compile Caffe
cp Makefile.config.example Makefile.config
# For some reason, I was getting <mpi.h> not found. So I had to manually edit /usr/include/H5public.h and disable PARALLEL support. (by adding #undef H5_HAVE_PARALLEL )
# Also, I disabled CUDNN (because these are pure-caffe benchmarks)
make all
make test
make runtest
