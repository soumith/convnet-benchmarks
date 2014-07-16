Build ccv
--------
```
sudo apt-get install libgsl0-dev

git clone https://github.com/liuliu/ccv.git
cd ccv
cd lib
./configure
cd ../bin/cuda
make
cd ../../../
```

Run cwc-bench
-------------
ccv/bin/cuda/cwc-bench list.txt
