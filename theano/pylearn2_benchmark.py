import os
import sys
import numpy as np
try:
    import theano.misc.pycuda_init
    import pycuda.driver
except ImportError:
    print "Note: pycuda not available, no timing via CUDA events possible"
    import time
    pycuda = None
import theano

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

def benchmark_three_ways(name, sharedX, sharedY, sharedW, X, Y, gW, gX, flops, mode=None):
    # benchmark fprop
    try:
        fprop = theano.function([], [],
                                givens=[(X, sharedX)],
                                updates=[(sharedY, Y)],
                                mode=mode)
        tm = time_run(fprop)
        del fprop
        print name, 'fprop:', (flops[0] / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'
    except Exception:
        print name, 'fprop: FAILED'

    # benchmark bprop wrt weights
    try:
        bprop = theano.function([], [],
                                givens=[(X, sharedX)],
                                updates=[(sharedW, gW)],
                                mode=mode)
        tm = time_run(bprop)
        del bprop
        print name, 'bprop weights:', (flops[1] / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'
    except Exception:
        print name, 'bprop weights: FAILED'

    # benchmark bprop wrt input
    try:
        bprop = theano.function([], [],
                                updates=[(sharedX, gX)],
                                mode=mode)
        tm = time_run(bprop)
        del bprop
        print name, 'bprop inputs:', (flops[2] / tm / 1e9), 'GFLOP/s ( tm =', tm, ')'
    except Exception:
        print name, 'bprop inputs: FAILED'

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

for run in runs:
    # params for run:
    # (input channels, output channels, kernel width, kernel height, batchsize, image width, image height, horizontal stride, vertical stride)
    ni, no, kw, kh, bs, iw, ih, dw, dh = run['ni'], run['no'], run['kw'], run['kh'], run['bs'], run['iw'], run['ih'], run['dw'], run['dh']
    print ''
    print 'CONFIG: input =', ni, 'x', iw, 'x', ih, '* ker =', ni, 'x', no, 'x', kw, 'x', kh, '( bs =', bs, ', stride =', dw, ')'
    ops = 2  # ops per point
    fprop_flops = ni*no*kw*kh*(iw-kw+1)*(ih-kh+1) / dw/dh * bs * ops  # flops of a valid convolution of X with W
    bprop_W_flops = 0 #no*(iw-kw+1)*(ih-kh+1)*ops*ni*kw*kh*no/dw/dh  # TODO: check if that's the flops of a valid convolution of X with Y
    bprop_X_flops = 0 #fprop_flops  # TODO: check if that's the flops of a full convolution of Y with W
    flops = (fprop_flops, bprop_W_flops, bprop_X_flops)

    # benchmark Theano standard convolution
    input_shape = (bs, ni, ih, iw)
    filter_shape = (no, ni, kh, kw)
    sharedX = theano.shared(np.random.randn(*input_shape).astype('float32'))
    sharedY = theano.shared(np.random.randn(bs, no, (ih-kh)/dh+1, (iw-kw)/dw+1).astype('float32'))
    sharedW = theano.shared(np.random.randn(*filter_shape).astype('float32'))
    X = theano.tensor.tensor4()
    Y = theano.tensor.nnet.conv.conv2d(X, sharedW, input_shape, filter_shape, subsample=(dh,dw))
    gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
    gX = theano.grad(None, wrt=X, known_grads={Y: sharedY})
    if int(os.environ.get("SKIP_LEGACY", 0)) == 0:
        benchmark_three_ways('theano.tensor.nnet.conv.conv2d',
                             sharedX, sharedY, sharedW, X, Y, gW, gX, flops)

    # benchmark Theano FFT convolution
    # Mimic Theano flag THEANO_FLAGS=optimizer_including=conv_fft_valid:conv_fft_full
    mode = theano.compile.get_default_mode()
    mode = mode.including('conv_fft_valid', 'conv_fft_full')
    benchmark_three_ways('(experimental) theano.sandbox.cuda.fftconv.conv2d_fft',
                         sharedX, sharedY, sharedW, X, Y, gW, gX, flops, mode)

    # benchmark caffe-like gemm convolution
    # Mimic Theano flag THEANO_FLAGS=optimizer_including=conv_gemm
    mode = theano.compile.get_default_mode()
    mode = mode.including('conv_gemm')
    benchmark_three_ways('(experimental, auto) theano.sandbox.cuda.blas.GpuCorrMM',
                         sharedX, sharedY, sharedW, X, Y, gW, gX, flops, mode)

    # benchmark caffe-like gemm convolution again, directly, w/o kernel flipping
    Y = theano.sandbox.cuda.blas.GpuCorrMM(subsample=(dh,dw))(gpu_contiguous(X), gpu_contiguous(sharedW))
    gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
    gX = theano.grad(None, wrt=X, known_grads={Y: sharedY})
    benchmark_three_ways('(experimental, manual) theano.sandbox.cuda.blas.GpuCorrMM',
                         sharedX, sharedY, sharedW, X, Y, gW, gX, flops, mode)

    del sharedX
    del sharedY
    del sharedW

    # benchmark cuda-convnet convolution
    # we use the pylearn2 wrapper for cuda-convnet (http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html)
    if FilterActs is None:
        continue  # skip cuda-convnet if pylearn2 wrapper is not available
    #(channels, rows, columns, batch_size)
    inputBatch = np.random.randn(ni, ih, iw, bs)
    sharedX = theano.shared(inputBatch.astype('float32'))
    sharedY = theano.shared(np.random.randn(no, (ih-kh)/dh+1, (iw-kw)/dw+1, bs).astype('float32'))
    # (channels, rows, columns, number of filters)
    sharedW = theano.shared(np.random.randn(ni, kh, kw, no).astype('float32'))
    contiguous_input = gpu_contiguous(sharedX)
    contiguous_filters = gpu_contiguous(sharedW)
    Y = FilterActs()(contiguous_input, contiguous_filters)
    gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
    #from pylearn2.sandbox.cuda_convnet.weight_acts import WeightActs
    #gW = WeightActs()(contiguous_input, gpu_contiguous(sharedY), sharedW.shape)[0]  # constructed by hand, results in the same graph
    gX = theano.grad(None, wrt=sharedX, known_grads={Y: sharedY})
    benchmark_three_ways('pylearn2.sandbox.cuda_convnet',
                         sharedX, sharedY, sharedW, X, Y, gW, gX, flops)

    del sharedX
    del sharedY
    del sharedW
