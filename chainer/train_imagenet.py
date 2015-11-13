#!/usr/bin/env python
import argparse
import time

import numpy as np

from chainer import cuda
from chainer import optimizers

parser = argparse.ArgumentParser(
    description=' convnet benchmarks on imagenet')
parser.add_argument('--arch', '-a', default='alex',
                    help='Convnet architecture \
                    (nin, alex, alexbn, googlenet, googlenetbn)')
parser.add_argument('--batchsize', '-B', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

# Prepare model
if args.arch == 'alex':
    import alex
    model = alex.Alex()
elif args.arch == 'googlenet':
    import googlenet
    model = googlenet.GoogLeNet()
else:
    raise ValueError('Invalid architecture name')

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(model)


def train_loop():
    # Trainer
    data = np.ndarray((args.batchsize, 3, model.insize, model.insize), dtype=np.float32)
    data.fill(33333)

    label = np.ndarray((args.batchsize), dtype=np.int32)
    label.fill(1)
    for i in range(10):
        print "Iteration", i

        x = xp.asarray(data)
        y = xp.asarray(label)

        optimizer.zero_grads()
        start = time.clock()
        loss, accuracy = model.forward(x, y)
        end = time.clock()
        print "Forward step time elapsed:", end - start, "seconds"

        start = time.clock()
        loss.backward()
        end = time.clock()
        print "Backward step time elapsed:", end - start, "seconds"

        start = time.clock()
        optimizer.update()
        end = time.clock()
        print "Optimizer step time elapsed:", end - start, "seconds"

        del loss, accuracy

train_loop()
