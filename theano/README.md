Install Theano:
```
git clone git://github.com/Theano/Theano.git
cd Theano
sudo python setup.py develop
```

Install pylearn2:
```
git clone git://github.com/lisa-lab/pylearn2.git
cd pylearn2
sudo python setup.py develop
```

Launch the script:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python pylearn2_benchmark.py 
```
