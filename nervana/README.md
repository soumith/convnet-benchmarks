Nervana Systems provided me with limited-release beta kernels for benchmarking (and correctness-checking).

They are open to releasing them publicly, but just dont have the bandwidth to support the open-release.
So, if you would like to have the kernels, email them: http://www.nervanasys.com/

The kernels come similarly packaged to their https://github.com/NervanaSystems/nervana-lib-gpu-performance-preview



SETUP
=====
git clone git@github.com:NervanaSystems/maxas.git
cd maxas
perl Makefile.PL
make
sudo make install

cd ..
git clone -b convnew git@github.com:NervanaSystems/nervanagpu.git
cd nervanagpu
make kernels
make python

cd ..
python nervanagpu/benchmarks/convnet-benchmarks.py |tee output.log


I am copying over the convnet-benchmarks.py to this folder for public viewing, just to make sure that there's nothing funny in the benchmarking itself and that it's clear that the benchmarking is done properly.

