Install Theano:
```
git clone git://github.com/Theano/Theano.git
cd Theano
python setup.py develop
# To install into your home directory instead:
# python setup.py develop --prefix=~/.local
```

Install pylearn2:
```
git clone git://github.com/lisa-lab/pylearn2.git
cd pylearn2
python setup.py develop
# To install into your home directory instead:
# python setup.py develop --prefix=~/.local
```

Install pycuda:
```
wget -c https://pypi.python.org/packages/source/p/pycuda/pycuda-2013.1.1.tar.gz#md5=acf9319ab2970d9700ed6486aa87b708
tar -xvf pycuda-2013.1.1.tar.gz
cd pycuda-2013.1.1
./configure.py
python setup.py install
# To install into your home directory instead:
# python setup.py install --user
```

Install scikits.cuda:
```
git clone https://github.com/lebedov/scikits.cuda.git
cd scikits.cuda
python setup.py install
# To install into your home directory instead:
# python setup.py install --user
```

Launch the script:
```
SKIP=legacy python pylearn2_benchmark.py
```
