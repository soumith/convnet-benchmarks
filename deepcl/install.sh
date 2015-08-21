#!/bin/bash

sudo apt-get install -y python2.7 python2.7-dev python-virtualenv cmake cmake-curses-gui make g++ gcc git
git clone --recursive https://github.com/hughperkins/DeepCL.git -b soumith-benchmarks
cd DeepCL
mkdir build
cd build
cmake ..
make -j 4 install
cd ../python
virtualenv ../env
source ../env/bin/activate
source ../dist/bin/activate.sh
python setup.py install
cd ../..

