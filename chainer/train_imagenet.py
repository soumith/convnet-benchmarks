#!/usr/bin/env python
import argparse
import datetime
import random
import sys
import time

import numpy as np

from chainer import computational_graph as c
from chainer import cuda
from chainer import optimizers

parser = argparse.ArgumentParser(
    description=' convnet benchmarks on imagenet')
parser.add_argument('--arch', '-a', default='alexbn',
                    help='Convnet architecture \
                    (nin, alexbn, googlenet, googlenetbn)')
parser.add_argument('--batchsize', '-B', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

# Prepare model
if args.arch == 'alexbn':
    import alexbn
    model = alexbn.AlexBN()
elif args.arch == 'googlenet':
    import googlenet
    model = googlenet.GoogLeNet()
elif args.arch == 'googlenetbn':
    import googlenetbn
    model = googlenetbn.GoogLeNetBN()
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
    graph_generated = False
    while True:
        x = xp.asarray(inp[0])
        y = xp.asarray(inp[1])

        optimizer.zero_grads()
        loss, accuracy = model.forward(x, y)
        loss.backward()
        optimizer.update()

        if not graph_generated:
            with open('graph.dot', 'w') as o:
                o.write(c.build_computational_graph((loss,), False).dump())
            with open('graph.wo_split.dot', 'w') as o:
                o.write(c.build_computational_graph((loss,), True).dump())
            print('generated graph')
            graph_generated = True

        
        del loss, accuracy, x, y

train_loop()
