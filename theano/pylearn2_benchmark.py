import time
import gc
import numpy as np
import theano
import pylearn2

import theano.tensor as T

from pylearn2.models.mlp import ConvElemwise, ConvNonlinearity, MLP
from pylearn2.space import Conv2DSpace

from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous


steps = 4  # nb of steps in loop to average perf
ops = 2  # ops per point

runs = [
   {
      'ni': 3,
      'no': 96,
      'kw': 11,
      'kh': 11,
      'iw': 128,
      'ih': 128,
      'bs': 128,
      'dw': 1,
      'dh': 1,
   },
   {
      'ni': 64,
      'no': 128,
      'kw': 9,
      'kh': 9,
      'iw': 64,
      'ih': 64,
      'bs': 128,
      'dw': 1,
      'dh': 1,
   },
   {
      'ni': 128,
      'no': 128,
      'kw': 9,
      'kh': 9,
      'iw': 32,
      'ih': 32,
      'bs': 128,
      'dw': 1,
      'dh': 1,
   },
   {
      'ni': 128,
      'no': 128,
      'kw': 7,
      'kh': 7,
      'iw': 16,
      'ih': 16,
      'bs': 128,
      'dw': 1,
      'dh': 1,
   }
]

for i in range(4):
    run = runs[i]
    # params for run:
    ni, no, kw, kh, bs, iw, ih, dw, dh = run['ni'], run['no'], run['kw'], run['kh'], run['bs'], run['iw'], run['ih'], run['dw'], run['dh']
    print ''
    print 'CONFIG: input =', ni, 'x', iw, 'x', ih, '* ker =', ni, 'x', no, 'x', kw, 'x', kh, '( bs =', bs, ', stride =', dw, ')'

    conv = MLP(
       batch_size=bs,
       input_space=Conv2DSpace((ih, iw), num_channels=ni, axes=('b', 'c', 0, 1)),
       layers=[ConvElemwise(no, (kw, kh), 'ConvTest', ConvNonlinearity(), irange=0.1)]
    )

    inputBatch = np.random.randn(bs, ni, ih, iw)
    sharedX = theano.shared(inputBatch.astype('float32'))
    sharedY = theano.shared(np.random.randn(bs, no, (ih-kh)/dh+1, (iw-kw)/dw+1).astype('float32'))

    X = theano.tensor.tensor4()

    Y = conv.fprop(X)

    fprop = theano.function([], [], givens=[(X, sharedX)], updates=[(sharedY, Y)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        fprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    print 'pylearn2.models.mlp.ConvElemwise:', (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) / dw/dh * bs * ops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    # Mimic Theano flag THEANO_FLAGS=optimizer_including=conv_fft_valid:conv_fft_full
    mode = theano.compile.get_default_mode()
    mode = mode.including('conv_fft_valid', 'conv_fft_full')
    fprop = theano.function([], [], givens=[(X, sharedX)],
                            updates=[(sharedY, Y)],
                            on_unused_input='ignore', mode=mode)

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        fprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del fprop
    del sharedX
    del conv
    del sharedY

    print '(fft experimental) pylearn2.models.mlp.ConvElemwise:', (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) / dw/dh * bs * ops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    ### pylearn2 work-around for using cuda-convnet (http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html) ###

    #(channels, rows, columns, batch_size)
    inputBatch = np.random.randn(ni, ih, iw, bs)
    sharedX = theano.shared(inputBatch.astype('float32'))
    sharedY = theano.shared(np.random.randn(no, (ih-kh)/dh+1, (iw-kw)/dw+1, bs).astype('float32'))
    # (channels, rows, columns, number of filters)
    sharedW = theano.shared(np.random.randn(ni, kh, kw, no).astype('float32'))

    conv_op = FilterActs()
    contiguous_input = gpu_contiguous(sharedX)
    contiguous_filters = gpu_contiguous(sharedW)
    Y = conv_op(contiguous_input, contiguous_filters)

    fprop = theano.function([], [], givens=[(X, sharedX)], updates=[(sharedY, Y)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        fprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del fprop
    del sharedX
    del conv_op
    del sharedY
    del sharedW

    print ' pylearn2.sandbox.cuda_convnet:', (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) / dw/dh * bs * ops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'
