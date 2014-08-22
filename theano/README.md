Install Theano:
```
git clone git://github.com/Theano/Theano.git
cd Theano
git pull https://github.com/f0k/Theano corrmm-faster-fullconv
sudo python setup.py develop
```

Install pylearn2:
```
git clone git://github.com/lisa-lab/pylearn2.git
cd pylearn2
sudo python setup.py develop
```

Install pycuda:
```
wget -c https://pypi.python.org/packages/source/p/pycuda/pycuda-2013.1.1.tar.gz#md5=acf9319ab2970d9700ed6486aa87b708
tar -xvf pycuda-2013.1.1.tar.gz
cd pycuda-2013.1.1
./configure.py
sudo python setup.py install
```

Install scikits.cuda:
```
git clone https://github.com/lebedov/scikits.cuda.git
cd scikits.cuda
sudo python setup.py install
```

Launch the script:
```
SKIP_LEGACY=1 THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python pylearn2_benchmark.py
```
