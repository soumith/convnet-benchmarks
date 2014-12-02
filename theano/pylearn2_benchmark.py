import os
import sys
import numpy as np
import math

import theano
if not theano.config.device.startswith('gpu'):
    import theano.sandbox.cuda
    theano.sandbox.cuda.use('gpu')
theano.config.floatX = 'float32'

try:
    import theano.misc.pycuda_init
    import pycuda.driver
except ImportError:
    print "Note: pycuda not available, no timing via CUDA events possible"
    import time
    pycuda = None
import theano

try:
    import theano.sandbox.cuda.dnn
    if not theano.sandbox.cuda.dnn.dnn_available():
        del theano.sandbox.cuda.dnn
        raise ImportError
except (ImportError, NameError):
    print "Note: cuDNN not available"

try:
    from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
except ImportError:
    FilterActs = None
    print "Note: pylearn2's cuda-convnet wrapper not available"
else:
    from theano.sandbox.cuda.basic_ops import gpu_contiguous


number = 10  # nb of steps in loop to average over
repeat = 1   # nb of trials to pick the minimum of

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

def time_run(fn):
    times = []
    fn()  # warm-up call, not timed
    if pycuda:
        theano.sandbox.cuda.synchronize()
        start = pycuda.driver.Event()
        end = pycuda.driver.Event()
        for _ in range(repeat):
            start.record()
            for _ in range(number):
                fn()
            end.record()
            end.synchronize()
            times.append(start.time_till(end) / 1e3 / number)
    else:
        for _ in range(repeat):
            theano.sandbox.cuda.synchronize()
            start = time.time()
            for _ in range(number):
                fn()
            theano.sandbox.cuda.synchronize()
            times.append((time.time() - start) / number)
    return min(times)

def print_graph(fn):
    if int(os.environ.get('PRINT_GRAPH', 0)):
		# debugprint of graph (in blue text)
		print '\033[1;34m'
		theano.printing.debugprint(fn)
		print '\033[1;m'

def benchmark_three_ways(name, sharedX, sharedY, sharedW, X, Y, gW, gX, mode=None):
    # benchmark fprop
    try:
        fprop = theano.function([], [],
                                givens=[(X, sharedX)],
                                updates=[(sharedY, Y)],
                                mode=mode,
                                name=name + " fprop")
        tm = time_run(fprop)
        print '{: <50} ==> {: <13} ==> {: >7}'.format(name, 'fprop', int(tm*1000))
        print_graph(fprop)
        del fprop
    except Exception, e:
        print name, 'fprop: FAILED', str(e).split('\n', 1)[0]

    # benchmark bprop wrt input
    try:
        bprop = theano.function([], [],
                                # the nvidia wrapper need this (in fact could be optional for subsample==(1, 1)
                                givens=[(X, sharedX)],
                                updates=[(sharedX, gX)],
                                mode=mode,
                                name=name + " bprop inputs")
        tm = time_run(bprop)
        print '{: <50} ==> {: <13} ==> {: >7}'.format(name, 'bprop inputs', int(tm*1000))
        print_graph(bprop)
        del bprop
    except Exception, e:
        print name, 'bprop inputs: FAILED', str(e).split('\n', 1)[0]

    # benchmark bprop wrt weights
    try:
        bprop = theano.function([], [],
                                givens=[(X, sharedX)],
                                updates=[(sharedW, gW)],
                                mode=mode,
                                name=name + " bprop weights")
        tm = time_run(bprop)
        print '{: <50} ==> {: <13} ==> {: >7}'.format(name, 'bprop weights', int(tm*1000))
        print_graph(bprop)
        del bprop
    except Exception, e:
        print name, 'bprop weights: FAILED', str(e).split('\n', 1)[0]
    print ''

def parse_custom_config(s):
    # parses a custom configuration string of the format:
    # iAxBxC,kDxExF,bG,sHxJ where A: input channels, B: input width, C: input height,
    # D: output channels, E: kernel width, F: kernel height, G: batchsize,
    # H: horizontal stride, J: vertical stride (with G, H, J being optional)
    run = {'bs': 128, 'dw': 1, 'dh': 1}
    defs = {'i': ['ni', 'iw', 'ih'],
            'k': ['no', 'kw', 'kh'],
            'b': ['bs'],
            's': ['dw', 'dh']}
    for part in s.split(','):
        p, args = part[0], map(int, part[1:].split('x'))
        run.update(zip(defs[p], args))
    return run

if len(sys.argv) > 1:
    # allow specifying the runs on command line, 1-indexed (i.e., 1 2 5)
    runs = [runs[int(r) - 1] for r in sys.argv[1:] if r[0] != 'i']
    # allow specifying custom configurations on command line (e.g., i3x80x15,k32x3x7,b256)
    runs.extend([parse_custom_config(r) for r in sys.argv[1:] if r[0] == 'i'])

# allow specifying benchmarks to skip via a SKIP environment variable
skip_tests = os.environ.get("SKIP", '').lower().split(',')

for run in runs:
    # params for run:
    # (input channels, output channels, kernel width, kernel height, batchsize, image width, image height, horizontal stride, vertical stride)
    ni, no, kw, kh, bs, iw, ih, dw, dh = run['ni'], run['no'], run['kw'], run['kh'], run['bs'], run['iw'], run['ih'], run['dw'], run['dh']
    print ''
    print 'CONFIG: input =', ni, 'x', iw, 'x', ih, '* ker =', ni, 'x', no, 'x', kw, 'x', kh, '( bs =', bs, ', stride =', dw, ')'
    ops = 2  # ops per point
    mode = theano.compile.get_default_mode()

    # benchmark Theano legacy convolution
    # Mimic THEANO_FLAGS=optimizer_excluding=conv_gemm:conv_dnn
    input_shape = (bs, ni, ih, iw)
    filter_shape = (no, ni, kh, kw)
    try:
        sharedX = theano.shared(np.random.randn(*input_shape).astype('float32'), name='sharedX')
        sharedY = theano.shared(np.random.randn(bs, no, (ih-kh)/dh+1, (iw-kw)/dw+1).astype('float32'), name='sharedY')
        sharedW = theano.shared(np.random.randn(*filter_shape).astype('float32'), name='sharedW')
    except MemoryError, e:
        print "SKIPPING config due to the memory error below"
        print e
        continue
    X = theano.tensor.tensor4('X')
    Y = theano.tensor.nnet.conv.conv2d(X, sharedW, input_shape, filter_shape, subsample=(dh,dw))
    gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
    gX = theano.grad(None, wrt=X, known_grads={Y: sharedY})
    if 'legacy' not in skip_tests:
        benchmark_three_ways('theano.tensor.nnet.conv.conv2d',
                             sharedX, sharedY, sharedW, X, Y, gW, gX,
                             mode.excluding('conv_gemm', 'conv_dnn'))

    # benchmark Theano meta-optimizer
    # Mimic THEANO_FLAGS=optimizer_including=conv_meta
    if 'meta' not in skip_tests:
        benchmark_three_ways('(experimental) meta-optimizer',
                             sharedX, sharedY, sharedW, X, Y, gW, gX,
                             mode.including('conv_meta'))

    # benchmark Theano FFT convolution
    # Mimic THEANO_FLAGS=optimizer_including=conv_fft
    if 'fft' not in skip_tests:
        benchmark_three_ways('theano.sandbox.cuda.fftconv.conv2d_fft',
                             sharedX, sharedY, sharedW, X, Y, gW, gX,
                             mode.including('conv_fft'))

    # benchmark cudnn, convolution with kernel flipping
    if hasattr(theano.sandbox.cuda, 'dnn') and 'dnn' not in skip_tests:
        benchmark_three_ways('(auto) theano.sandbox.cuda.dnn.GpuDnnConv',
	                         sharedX, sharedY, sharedW, X, Y, gW, gX,
	                         mode.including('conv_dnn'))

    # benchmark caffe-like gemm convolution
    # Mimic THEANO_FLAGS=optimizer_excluding=conv_dnn
    if 'gemm' not in skip_tests and 'caffe' not in skip_tests:
        benchmark_three_ways('(auto) theano.sandbox.cuda.blas.GpuCorrMM',
                             sharedX, sharedY, sharedW, X, Y, gW, gX,
                             mode.excluding('conv_dnn'))

        # benchmark caffe-like gemm convolution again, directly, w/o kernel flipping
        Y = theano.sandbox.cuda.blas.GpuCorrMM(subsample=(dh, dw))(
            gpu_contiguous(X), gpu_contiguous(sharedW))
        gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
        gX = theano.grad(None, wrt=X, known_grads={Y: sharedY})
        benchmark_three_ways('(manual) theano.sandbox.cuda.blas.GpuCorrMM',
                             sharedX, sharedY, sharedW, X, Y, gW, gX)

    # benchmark nvidia convolution directly
    if hasattr(theano.sandbox.cuda, 'dnn') and 'dnn' not in skip_tests:
        Y = theano.sandbox.cuda.dnn.dnn_conv(X, sharedW, 'valid',
                                             subsample=(dh, dw))
        gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
        gX = theano.grad(None, wrt=X, known_grads={Y: sharedY})
        benchmark_three_ways(
            '(manual conv) theano.sandbox.cuda.dnn.GpuDnnConv',
            sharedX, sharedY, sharedW, X, Y, gW, gX)
    if int(os.environ.get('DNN_CORR', 0)):
        # without flipping (just as fast as manual conv; set DNN_CORR=1 to run)
        Y = theano.sandbox.cuda.dnn.dnn_conv(X, sharedW, 'valid',
                                             subsample=(dh, dw),
                                             conv_mode='cross')
        gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
        gX = theano.grad(None, wrt=X, known_grads={Y: sharedY})
        benchmark_three_ways(
            '(manual corr) theano.sandbox.cuda.dnn.GpuDnnConv',
            sharedX, sharedY, sharedW, X, Y, gW, gX)

    del sharedX
    del sharedY
    del sharedW

    # benchmark cuda-convnet convolution
    # we use the pylearn2 wrapper for cuda-convnet (http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html)
    if (FilterActs is None) or ('convnet' in skip_tests):
        continue  # skip cuda-convnet if pylearn2 wrapper is not available
    #(channels, rows, columns, batch_size)
    inputBatch = np.random.randn(ni, ih, iw, bs)
    sharedX = theano.shared(inputBatch.astype('float32'))
    sharedY = theano.shared(np.random.randn(no, (ih-kh)/dh+1, (iw-kw)/dw+1, bs).astype('float32'))
    # (channels, rows, columns, number of filters)
    sharedW = theano.shared(np.random.randn(ni, kh, kw, no).astype('float32'))
    contiguous_input = gpu_contiguous(sharedX)
    contiguous_filters = gpu_contiguous(sharedW)
    for partial_sum in (None, 1):
        Y = FilterActs(partial_sum=partial_sum)(contiguous_input, contiguous_filters)
        gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
        gX = theano.grad(None, wrt=sharedX, known_grads={Y: sharedY})
        benchmark_three_ways('pylearn2.sandbox.cuda_convnet(partial_sum=%r)' % partial_sum,
                             sharedX, sharedY, sharedW, X, Y, gW, gX)

    del sharedX
    del sharedY
    del sharedW
