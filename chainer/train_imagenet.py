#!/usr/bin/env python
import argparse
import time

import numpy as np

from chainer import cuda
from chainer import optimizers


parser = argparse.ArgumentParser(
    description=' convnet benchmarks on imagenet')
parser.add_argument('--arch', '-a', default='alexnet',
                    help='Convnet architecture \
                    (alex, googlenet, vgga, overfeat)')
parser.add_argument('--batchsize', '-B', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

# Prepare model
print(args.arch)
if args.arch == 'alexnet':
    import alex
    model = alex.Alex()
elif args.arch == 'googlenet':
    import googlenet
    model = googlenet.GoogLeNet()
elif args.arch == 'vgga':
    import vgga
    model = vgga.vgga()
elif args.arch == 'overfeat':
    import overfeat
    model = overfeat.overfeat()
else:
    raise ValueError('Invalid architecture name')

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(model)

workspace_size = int(1 * 2**30)
import chainer

chainer.cuda.set_max_workspace_size(workspace_size)

chainer.config.train = True
chainer.config.use_cudnn = 'always'


class Timer():
    def preprocess(self):
        if xp == np:
            self.start = time.time()
        else:
            self.start = xp.cuda.Event()
            self.end = xp.cuda.Event()
            self.start.record()

    def postprocess(self):
        if xp == np:
            self.end = time.time()
        else:
            self.end.record()
            self.end.synchronize()

    def getElapseTime(self):
        if xp == np:
            return (self.end - self.start) * 1000
        else:
            return xp.cuda.get_elapsed_time(self.start, self.end)


def train_loop():
    # Trainer
    data = np.ndarray((args.batchsize, 3, model.insize,
                       model.insize), dtype=np.float32)
    data.fill(33333)
    total_forward = 0
    total_backward = 0
    niter = 13
    n_dry = 3

    label = np.ndarray((args.batchsize), dtype=np.int32)
    label.fill(1)
    count = 0
    timer = Timer()
    for i in range(niter):
        x = xp.asarray(data)
        y = xp.asarray(label)

        if args.arch == 'googlenet':
            timer.preprocess()
            out1, out2, out3 = model.forward(x)
            timer.postprocess()
            time_ = timer.getElapseTime()
            if i > n_dry - 1:
                count += 1
                total_forward += time_
            out = out1 + out2 + out3
        else:
            timer.preprocess()
            out = model.forward(x)
            timer.postprocess()
            time_ = time_ = timer.getElapseTime()
            if i > n_dry - 1:
                count += 1
                total_forward += time_

        out.zerograd()
        out.grad.fill(3)
        model.cleargrads()
        if xp != np:
            xp.cuda.Stream(null=True)
        timer.preprocess()
        out.backward()
        timer.postprocess()
        time_ = timer.getElapseTime()
        if i > n_dry - 1:
            total_backward += time_
        model.cleargrads()

        del out, x, y
        if args.arch == 'googlenet':
            del out1, out2, out3
    print("Average Forward:  ", total_forward / count, " ms")
    print("Average Backward: ", total_backward / count, " ms")
    print("Average Total:    ", (total_forward + total_backward) / count, " ms")
    print("")


train_loop()
