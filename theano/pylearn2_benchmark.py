import time
import gc
import numpy as np
import theano
import pylearn2

import theano.tensor as T

from pylearn2.models.mlp import ConvElemwise, ConvNonlinearity, MLP
from pylearn2.space import Conv2DSpace

from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.weight_acts import WeightActs
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
   },
   {
      'ni': 384,
      'no': 384,
      'kw': 3,
      'kh': 3,
      'iw': 13,
      'ih': 13,
      'bs': 128,
      'dw': 1,
      'dh': 1,
   }
]

for i in range(5):
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
    sharedW = conv.layers[0].transformer.get_params()[0]

    X = theano.tensor.tensor4()
    Y = conv.fprop(X)
    gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})

    # benchmark fprop Theano standard convolution
    fprop = theano.function([], [], givens=[(X, sharedX)], updates=[(sharedY, Y)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        fprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del fprop
    print 'pylearn2.models.mlp.ConvElemwise fprop:', (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) / dw/dh * bs * ops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    # benchmark bprop Theano standard convolution
    bprop = theano.function([], [], givens=[(X, sharedX)], updates=[(sharedW, gW)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        bprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del bprop
    print 'pylearn2.models.mlp.ConvElemwise bprop:', (0000 / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'  # TODO: GFLOP/s


    # Mimic Theano flag THEANO_FLAGS=optimizer_including=conv_fft_valid:conv_fft_full
    mode = theano.compile.get_default_mode()
    mode = mode.including('conv_fft_valid', 'conv_fft_full')

    # benchmark fprop Theano FFT convolution
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
    print '(fft experimental) pylearn2.models.mlp.ConvElemwise fprop:', (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) / dw/dh * bs * ops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    # benchmark bprop Theano FFT convolution
    bprop = theano.function([], [], givens=[(X, sharedX)],
                            updates=[(sharedW, gW)],
                            on_unused_input='ignore', mode=mode)

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        bprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del bprop
    print '(fft experimental) pylearn2.models.mlp.ConvElemwise bprop:', (0000 / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'  # TODO: GFLOP/s

    del sharedX
    del sharedY
    del sharedW
    del conv


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
    gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
    #gW = WeightActs()(contiguous_input, gpu_contiguous(sharedY), sharedW.shape)[0]  # constructed by hand, results in the same graph

    # benchmark fprop cuda-convnet convolution
    fprop = theano.function([], [], givens=[(X, sharedX)], updates=[(sharedY, Y)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        fprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del fprop
    print 'pylearn2.sandbox.cuda_convnet fprop:', (ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) / dw/dh * bs * ops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    # benchmark bprop cuda-convnet convolution
    bprop = theano.function([], [], givens=[(X, sharedX)], updates=[(sharedW, gW)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        bprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del bprop
    print 'pylearn2.sandbox.cuda_convnet bprop:', (0000 / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'  # TODO: GFLOP/s

    del sharedX
    del sharedY
    del sharedW
    del conv_op

