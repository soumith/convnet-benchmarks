#!/usr/bin/python

from __future__ import print_function

import os
import sys
import time
import array
import random
import PyDeepCL

numEpochs = 10

runs = [
   {
      'inputPlanes': 3,
      'outputPlanes': 96,
      'filterSize': 11,
      'inputSize': 128,
      'batchSize': 128,
   },
   {
      'inputPlanes': 64,
      'outputPlanes': 128,
      'filterSize': 9,
      'inputSize': 64,
      'batchSize': 128,
   },
   {
      'inputPlanes': 128,
      'outputPlanes': 128,
      'filterSize': 9,
      'inputSize': 32,
      'batchSize': 128,
   },
   {
      'inputPlanes': 128,
      'outputPlanes': 128,
      'filterSize': 7,
      'inputSize': 16,
      'batchSize': 128,
   },
   {
      'inputPlanes': 384, # num input planes
      'outputPlanes': 384, # num output planes
      'filterSize': 3, # filter size
      'inputSize': 13, # input size
      'batchSize': 128, # batchsize
   }
]

def time_run(fn):
    times = []
    fn()  # warm-up call, outputPlanest timed
    for _ in range(repeat):
        start = time.time()
        for _ in range(number):
            fn()
        times.append((time.time() - start) / number)
    return min(times)

def parse_custom_config(s):
    # parses a custom configuration string of the format:
    # iAxB,kCxD,bE where A: input channels, B: input size,
    # C: output channels, D: kernel size, E: batchsize,
    # (with G, being optional)
    run = {'batchSize': 128 }
    defs = {'i': ['inputPlanes', 'inputSize'],
            'k': ['outputPlanes', 'filterSize'],
            'b': ['batchSize'] }
    for part in s.split(','):
        p, args = part[0], map(int, part[1:].split('x'))
        run.update(zip(defs[p], args))
    return run

def go(runs):
    for run in runs:
        for key in run.keys(): # copy key values into function scope
            go.__globals__[key] = run[key]
        print( '' )
        print( 'CONFIG: ', run )

        net = PyDeepCL.NeuralNet( inputPlanes, inputSize )
        net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(inputPlanes)
            .filterSize(filterSize).biased().linear() )
        net.addLayer( PyDeepCL.SquareLossMaker() )
        print( net.asString() )

        images = array.array( 'f', [0] * (batchSize*inputPlanes*inputSize*inputSize) )
        grad = array.array('f',[0] * batchSize * outputPlanes * (inputSize - filterSize + 1) )
        for i in range( batchSize*inputPlanes*inputSize*inputSize ):
            images[i] = random.random() - 0.5
        for i in range( batchSize * outputPlanes * (inputSize - filterSize + 1) ):
            grad[i] = random.random() - 0.5
        
        try:
            net.setBatchSize(batchSize)
            forwardtime = 0
            backwardtime = 0
            last = time.time()
            for epoch in range(numEpochs): 
                net.propagate( images )
                now = time.time()
                forwardtime += (now-last)
                last = now
                net.backProp( 0.0001, grad )
                now = time.time()
                backwardtime += (now-last)
                last = now
            print('total forward time ', forwardtime, 'backwardtime',backwardtime,' total', (forwardtime + backwardtime))
            print('per epoch forward time ', forwardtime / numEpochs, 'backwardtime',backwardtime / numEpochs,' total', (forwardtime + backwardtime) / numEpochs)
            
        except Exception, e:
            print('something went wrong:', e )
            continue

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # allow specifying the runs on command line, 1-indexed (i.e., 1 2 5)
        runs = [runs[int(r) - 1] for r in sys.argv[1:] if r[0] != 'i']
        # allow specifying custom configurations on command line (e.g., i3x80x15,k32x3x7,b256)
        runs.extend([parse_custom_config(r) for r in sys.argv[1:] if r[0] == 'i'])

    go(runs)

