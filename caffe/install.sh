git clone https://github.com/BVLC/caffe.git
cd caffe


# Dependencies
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev 
sudo apt-get install -y protobuf-compiler gfortran libjpeg62 libfreeimage-dev libatlas-base-dev git python-dev python-pip 
sudo apt-get install -y libgoogle-glog-dev libbz2-dev libxml2-dev libxslt-dev libffi-dev libssl-dev libgflags-dev liblmdb-dev python-yaml
sudo easy_install pillow

mkdir build
cd build
export CMAKE_LIBRARY_PATH=/opt/OpenBLAS/include:/opt/OpenBLAS/lib:$CMAKE_LIBRARY_PATH
cmake .. -DCPU_ONLY=1 -DBLAS="open"
make
cd ../../

#wget -c https://software.intel.com/sites/default/files/managed/ad/a4/intel_optimized_technical_preview_for_multinode_caffe_1.0.tgz
#tar -xvf intel_optimized_technical_preview_for_multinode_caffe_1.0
#cd intel_optimized_technical_preview_for_multinode_caffe_1.0
#unzip -P accept intel_optimized_technical_preview_for_multinode_caffe_1.0.zip
