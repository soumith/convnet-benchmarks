import time
import gc
import numpy as np
import theano
import pylearn2

import theano.tensor as T

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
    # (input channels, output channels, kernel width, kernel height, batchsize, image width, image height, horizontal stride, vertical stride)
    ni, no, kw, kh, bs, iw, ih, dw, dh = run['ni'], run['no'], run['kw'], run['kh'], run['bs'], run['iw'], run['ih'], run['dw'], run['dh']
    print ''
    print 'CONFIG: input =', ni, 'x', iw, 'x', ih, '* ker =', ni, 'x', no, 'x', kw, 'x', kh, '( bs =', bs, ', stride =', dw, ')'
    fprop_flops = ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) / dw/dh * bs * ops  # flops of a valid convolution of X with W
    bprop_W_flops = 0 #no*(iw-kw+1)*(ih-kh+1)*ops*ni*kw*kh*no/dw/dh  # TODO: check if that's the flops of a valid convolution of X with Y
    bprop_X_flops = 0 #fprop_flops  # TODO: check if that's the flops of a full convolution of Y with W

    input_shape = (bs, ni, ih, iw)
    filter_shape = (no, ni, kh, kw)
    sharedX = theano.shared(np.random.randn(*input_shape).astype('float32'))
    sharedY = theano.shared(np.random.randn(bs, no, (ih-kh)/dh+1, (iw-kw)/dw+1).astype('float32'))
    sharedW = theano.shared(np.random.randn(*filter_shape).astype('float32'))

    X = theano.tensor.tensor4()
    Y = theano.tensor.nnet.conv.conv2d(X, sharedW, input_shape, filter_shape, subsample=(dh,dw))
    gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
    gX = theano.grad(None, wrt=X, known_grads={Y: sharedY})

    # benchmark Theano standard convolution, fprop
    fprop = theano.function([], [], givens=[(X, sharedX)], updates=[(sharedY, Y)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        fprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del fprop
    print 'theano.tensor.nnet.conv.conv2d fprop:', (fprop_flops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    # benchmark Theano standard convolution, bprop wrt weights
    bprop = theano.function([], [], givens=[(X, sharedX)], updates=[(sharedW, gW)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        bprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del bprop
    print 'theano.tensor.nnet.conv.conv2d bprop weights:', (bprop_W_flops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    # benchmark Theano standard convolution, bprop wrt input
    bprop = theano.function([], [], updates=[(sharedX, gX)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        bprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del bprop
    print 'theano.tensor.nnet.conv.conv2d bprop inputs:', (bprop_X_flops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    # benchmark Theano standard convolution, bprop wrt weights and input
    bprop = theano.function([], [], givens=[(X, sharedX)], updates=[(sharedW, gW), (sharedX, gX)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        bprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del bprop
    print 'theano.tensor.nnet.conv.conv2d bprop both:', ((bprop_W_flops + bprop_X_flops) / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'


    # Mimic Theano flag THEANO_FLAGS=optimizer_including=conv_fft_valid:conv_fft_full
    mode = theano.compile.get_default_mode()
    mode = mode.including('conv_fft_valid', 'conv_fft_full')

    # benchmark Theano FFT convolution, fprop
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
    print '(experimental) theano.sandbox.cuda.fftconv.conv2d_fft fprop:', (fprop_flops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    # benchmark Theano FFT convolution, bprop wrt weights
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
    print '(experimental) theano.sandbox.cuda.fftconv.conv2d_fft bprop weights:', (bprop_W_flops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    # benchmark Theano FFT convolution, bprop wrt inputs
    bprop = theano.function([], [],
                            updates=[(sharedX, gX)],
                            on_unused_input='ignore', mode=mode)

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        bprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del bprop
    print '(experimental) theano.sandbox.cuda.fftconv.conv2d_fft bprop inputs:', (bprop_X_flops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    del sharedX
    del sharedY
    del sharedW


    ### pylearn2 wrapper for using cuda-convnet (http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html) ###

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
    #from pylearn2.sandbox.cuda_convnet.weight_acts import WeightActs
    #gW = WeightActs()(contiguous_input, gpu_contiguous(sharedY), sharedW.shape)[0]  # constructed by hand, results in the same graph
    gX = theano.grad(None, wrt=sharedX, known_grads={Y: sharedY})

    # benchmark cuda-convnet convolution, fprop
    fprop = theano.function([], [], givens=[(X, sharedX)], updates=[(sharedY, Y)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        fprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del fprop
    print 'pylearn2.sandbox.cuda_convnet fprop:', (fprop_flops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    # benchmark cuda-convnet convolution, bprop wrt weights
    bprop = theano.function([], [], givens=[(X, sharedX)], updates=[(sharedW, gW)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        bprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del bprop
    print 'pylearn2.sandbox.cuda_convnet bprop weights:', (bprop_W_flops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    # benchmark cuda-convnet convolution, bprop wrt inputs
    bprop = theano.function([], [], updates=[(sharedX, gX)], on_unused_input='ignore')

    theano.sandbox.cuda.synchronize()
    start = time.time()
    for i in range(steps):
        bprop()
    theano.sandbox.cuda.synchronize()
    tm = (time.time()-start)/steps

    del bprop
    print 'pylearn2.sandbox.cuda_convnet bprop inputs:', (bprop_X_flops / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'

    del sharedX
    del sharedY
    del sharedW
    del conv_op

