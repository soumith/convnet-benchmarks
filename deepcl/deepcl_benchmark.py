#!/usr/bin/python

# This is intended to target inclusion in soumith's benchmarks as
# https://github.com/soumith/convnet-benchmarks

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

def writeResults( resultsLine ):
    f = open('results.txt', 'a')
    f.write( resultsLine + '\n' )
    f.close()

def time_layer( numEpochs, batchSize, inputPlanes, inputSize, outputPlanes, filterSize ):
    print('building network...')
    net = PyDeepCL.NeuralNet( inputPlanes, inputSize )
#    net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(inputPlanes)
#        .filterSize(1).padZeros().biased().linear() ) # this is just to make sure that gradient needs to be 
#                                                      # backproped through next layer
    net.addLayer( PyDeepCL.ForceBackpropMaker() ) # this forces the next layer to backprop gradients to
                          # this layer
    net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(outputPlanes)
        .filterSize(filterSize).biased().linear() )
    net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(1).imageSize(1) )
    net.addLayer( PyDeepCL.SoftMaxMaker() )
    print( net.asString() )

    images = array.array( 'f', [0] * (batchSize*inputPlanes*inputSize*inputSize) )
    for i in range( batchSize*inputPlanes*inputSize*inputSize ):
        images[i] = random.random() - 0.5
#    grad = array.array('f',[0] * batchSize * outputPlanes * (inputSize - filterSize + 1) )
#    for i in range( batchSize * outputPlanes * (inputSize - filterSize + 1) ):
#        grad[i] = random.random() - 0.5
    labels = array.array('i',[0] * batchSize )
    
    print('warming up...')
    #try:
    net.setBatchSize(batchSize)

    # warm up forward
    for i in range(8):
        last = time.time()
        net.propagate( images )
        now = time.time()
        print('  warm up propagate all-layer time', now - last )
        last = now
    net.backPropFromLabels( 0.001, labels )
    now = time.time()
    print('   warm up backprop all-layer time', now - last )
    last = now

    layer = net.getLayer(2)
    print('running forward prop timings:')
    for i in range(numEpochs):
        layer.propagate()
    now = time.time()
    print('forward layer total time', now - last )
    print('forward layer average time', ( now - last ) / float(numEpochs) )
    writeResults( layer.asString() + ', forward: ' + str( ( now - last ) / float(numEpochs) * 1000 ) + 'ms' )

    print('warm up backwards again')
    layer.backProp(0.001)
    layer.backProp(0.001)
    print('warm up backwards done. start timings:')

    now = time.time()
    last = now
    for i in range(numEpochs):
        layer.backProp(0.001)
    now = time.time()
    print('backwar layer total time', now - last )
    print('backwar layer average time', ( now - last ) / float(numEpochs) )
    writeResults( layer.asString() + ', backward: ' + str( ( now - last ) / float(numEpochs) * 1000 ) + 'ms' )
    last = now

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

        time_layer(numEpochs, batchSize, inputPlanes, inputSize,
            outputPlanes, filterSize )

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # allow specifying the runs on command line, 1-indexed (i.e., 1 2 5)
        runs = [runs[int(r) - 1] for r in sys.argv[1:] if r[0] != 'i']
        # allow specifying custom configurations on command line (e.g., i3x80x15,k32x3x7,b256)
        runs.extend([parse_custom_config(r) for r in sys.argv[1:] if r[0] == 'i'])

    go(runs)

