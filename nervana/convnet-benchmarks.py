#!/usr/bin/python

import numpy           as np
import pycuda.driver   as drv
from pycuda.autoinit   import context
from nervanagpu        import NervanaGPU
from nervanagpu.layers import DataLayer, ConvLayer, PoolLayer, FullLayer
print context.get_device().name()

# Compare results here:
# https://github.com/soumith/convnet-benchmarks

# number of full iterations
loops       = 10
# show bechmark details for each layer
layer_bench = 0
# show layer stats after each operation
print_stats = 0

ng = NervanaGPU(bench=layer_bench)

# don't learn, just benchmark
momentum      = 0.0
learning_rate = 0.0

# common convolutional layer settings
conv3   = { "R":3, "S":3, "pad_h":1, "pad_w":1 }
conv1   = { "R":1, "S":1, "pad_h":0, "pad_w":0 }

# traditional pooling
pool2   = { "op":"max", "R":2, "S":2 }
pool3   = { "op":"max", "R":3, "S":3, "str_h":2, "str_w":2 }

# maxout pooling
pool1j2 = { "op":"max", "J":2 } # maxout in the fc layers
pool2j2 = { "op":"max", "J":2, "R":2, "S":2 }
pool3j2 = { "op":"max", "J":2, "R":3, "S":3 }

networks = {
    "Alexnet" : (
        { "layer":DataLayer, "N":128, "C":3, "H":224, "W":224},
        { "layer":ConvLayer, "K":64, "R":11, "S":11, "str_h":4, "str_w":4, "pad_h":2, "pad_w":2 }, #, "update_size":"C128_K64"
        { "layer":PoolLayer, "common":pool3 },
        { "layer":ConvLayer, "K":192, "R":5, "S":5, "pad_h":2, "pad_w":2 },
        { "layer":PoolLayer, "common":pool3 },
        { "layer":ConvLayer, "K":384, "common":conv3 },
        { "layer":ConvLayer, "K":256, "common":conv3 },
        { "layer":ConvLayer, "K":256, "common":conv3 },
        { "layer":PoolLayer, "common":pool3 },
        { "layer":FullLayer, "nOut":3072 },
        { "layer":FullLayer, "nOut":3072 },
        { "layer":FullLayer, "nOut":1000 },
    ),
    "Overfeat" : (
        { "layer":DataLayer, "N":128, "C":3, "H":231, "W":231},
        { "layer":ConvLayer, "K":96, "R":11, "S":11, "str_h":4, "str_w":4 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":ConvLayer, "K":256, "R":5, "S":5 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":ConvLayer, "K":512,  "common":conv3 },
        { "layer":ConvLayer, "K":1024, "common":conv3 },
        { "layer":ConvLayer, "K":1024, "common":conv3 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":FullLayer, "nOut":3072 },
        { "layer":FullLayer, "nOut":3072 },
        { "layer":FullLayer, "nOut":1000 },
    ),
    # See http://arxiv.org/pdf/1409.1556.pdf for variations
    # Not all of these will fit on a 980, but forthcoming hardware is a diferent story.
    # The full VGG model fits at N=256 in 12GB with fp16.
    "VGG" : (
        { "layer":DataLayer, "N":64, "C":3, "H":224, "W":224},
        { "layer":ConvLayer, "K":64,  "common":conv3 },
       #{ "layer":ConvLayer, "K":64,  "common":conv3 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":ConvLayer, "K":128, "common":conv3 },
       #{ "layer":ConvLayer, "K":128, "common":conv3 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":ConvLayer, "K":256, "common":conv3 },
        { "layer":ConvLayer, "K":256, "common":conv3 },
       #{ "layer":ConvLayer, "K":256, "common":conv1 },
       #{ "layer":ConvLayer, "K":256, "common":conv3 },
       #{ "layer":ConvLayer, "K":256, "common":conv3 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
       #{ "layer":ConvLayer, "K":512, "common":conv1 },
       #{ "layer":ConvLayer, "K":512, "common":conv3 },
       #{ "layer":ConvLayer, "K":512, "common":conv3 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
       #{ "layer":ConvLayer, "K":512, "common":conv1 },
       #{ "layer":ConvLayer, "K":512, "common":conv3 },
       #{ "layer":ConvLayer, "K":512, "common":conv3 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":FullLayer, "nOut":3072 },
        { "layer":FullLayer, "nOut":3072 },
        { "layer":FullLayer, "nOut":1000 },
    ),
    # Here are some configs with Maxout pooling
    "Alexnet2" : (
        { "layer":DataLayer, "N":256, "C":3, "H":224, "W":224},
        { "layer":ConvLayer, "K":128, "R":11, "S":11, "str_h":4, "str_w":4, "pad_h":2, "pad_w":2 },
        { "layer":PoolLayer, "common":pool3j2 },
        { "layer":ConvLayer, "K":384, "R":5, "S":5, "pad_h":2, "pad_w":2 },
        { "layer":PoolLayer, "common":pool3j2 },
        { "layer":ConvLayer, "K":384, "common":conv3 },
        { "layer":ConvLayer, "K":256, "common":conv3 },
        { "layer":ConvLayer, "K":256, "common":conv3 },
        { "layer":PoolLayer, "common":pool3j2 },
        { "layer":FullLayer, "nOut":3072*2 },
        { "layer":PoolLayer, "common":pool1j2 },
        { "layer":FullLayer, "nOut":3072*2 },
        { "layer":PoolLayer, "common":pool1j2 },
        { "layer":FullLayer, "nOut":1000 },
    ),
    "Overfeat2" : (
        { "layer":DataLayer, "N":256, "C":3, "H":231, "W":231},
        { "layer":ConvLayer, "K":128, "R":11, "S":11, "str_h":4, "str_w":4 },
        { "layer":PoolLayer, "common":pool2j2 },
        { "layer":ConvLayer, "K":256, "R":5, "S":5 },
        { "layer":PoolLayer, "common":pool2j2 },
        { "layer":ConvLayer, "K":512,  "common":conv3 },
        { "layer":ConvLayer, "K":1024, "common":conv3 },
        { "layer":ConvLayer, "K":1024, "common":conv3 },
        { "layer":PoolLayer, "common":pool2j2 },
        { "layer":FullLayer, "nOut":3072*2 },
        { "layer":PoolLayer, "common":pool1j2 },
        { "layer":FullLayer, "nOut":3072*2 },
        { "layer":PoolLayer, "common":pool1j2 },
        { "layer":FullLayer, "nOut":1000 },
    ),
    "VGG2" : (
        { "layer":DataLayer, "N":64, "C":3, "H":224, "W":224},
        { "layer":ConvLayer, "K":64,  "common":conv3 },
        { "layer":ConvLayer, "K":64,  "common":conv3 }, #, "update_size":"C128_K128"
        { "layer":PoolLayer, "common":pool2 },
        { "layer":ConvLayer, "K":128, "common":conv3 },
        { "layer":ConvLayer, "K":128, "common":conv3 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":ConvLayer, "K":256, "common":conv3 },
        { "layer":ConvLayer, "K":256, "common":conv3 },
        #{ "layer":ConvLayer, "K":256, "common":conv1 },
        { "layer":ConvLayer, "K":256, "common":conv3 },
        { "layer":ConvLayer, "K":256, "common":conv3 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
        #{ "layer":ConvLayer, "K":512, "common":conv1 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
        #{ "layer":ConvLayer, "K":512, "common":conv1 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
        { "layer":ConvLayer, "K":512, "common":conv3 },
        { "layer":PoolLayer, "common":pool2 },
        { "layer":FullLayer, "nOut":3072 },
        #{ "layer":PoolLayer, "common":pool1j2 },
        { "layer":FullLayer, "nOut":3072 },
        #{ "layer":PoolLayer, "common":pool1j2 },
        { "layer":FullLayer, "nOut":1000 },
    ),
}

# "Alexnet","Overfeat","VGG", "Alexnet2","Overfeat2","VGG2"
for net in ("Alexnet","Overfeat","VGG",):

    for dtype in (np.float16, np.float32):

        network = networks[net]
        name    = "%s (dtype=%s, N=%d)" % (net, np.dtype(dtype).name, network[0]["N"])

        print "------------------------------------------------"
        print "Benchmarking: ", name
        print "------------------------------------------------"

        layers = []
        prev_layer = None
        max_deltas  = 0
        max_weights = 0
        max_delta_layer  = None
        max_weight_layer = None
        shared_weights   = None
        shared_deltas    = [0,0]

        for conf in network:

            config = dict(conf)

            config["dtype"] = dtype

            # merge shared params
            config.update(config.pop("common", {}))
            layer_type = config.pop("layer")

            #Propagate the fixed and calculated dimensions
            if prev_layer is not None:
                config["N"] = prev_layer.N

                # TODO: make propagating C=>K more elegent..
                if layer_type is FullLayer:
                    config["nIn"] = prev_layer.nOut
                elif layer_type is PoolLayer and type(prev_layer) is FullLayer:
                    config["C"] = prev_layer.nOut
                else:
                    config["C"] = prev_layer.K
                    config["H"] = prev_layer.P
                    config["W"] = prev_layer.Q

            # Instantiate the layer
            layer = layer_type(ng, **config)

            prev_layer = layer
            layers.append(layer)

            # find the size of the largest buffers so they can be shared
            if layer.sizeF > max_weights:
                max_weights = layer.sizeF
                max_weight_layer = layer

            if layer.sizeO > max_deltas:
                max_deltas = layer.sizeO
                max_delta_layer = layer

        # for layer in sorted(layers, key=lambda l: l.sizeO, reverse=True):
        #     print "%d %s" % (layer.sizeO, layer)

        # Init shared buffers (assumes consistent dtype for now)
        shared_deltas[0] = ng.empty(max_delta_layer.dimO2,  dtype=max_delta_layer.dtype)
        shared_deltas[1] = ng.empty(max_delta_layer.dimO2,  dtype=max_delta_layer.dtype)
        shared_weights   = ng.empty(max_weight_layer.dimF2, dtype=max_weight_layer.dtype)

        prev_layer = None
        delta = False
        for layer in layers:

            print layer

            # Intitalize buffers.  Alernate shared delta buffer.
            # One layer can't have the same buffer for both error in and error out.
            layer.init_activations()
            layer.init_weights(shared=shared_weights)
            layer.init_deltas(shared=shared_deltas[delta])

            # connect layer to previous layer
            layer.connect(prev_layer)
            prev_layer = layer
            delta = not delta

        remain, total = drv.mem_get_info()
        print "%.3fGB of %.3fGB Allocated (%.3fGB Remaining)" % ((total-remain)/1024.**3, total/1024.**3, remain/1024.**3)

        # give the first layer some data
        layers[0].init_data(np.random.uniform(0.0, 1.0, layers[0].dimO2))

        # Scale the initial weights so activations are bound around 1.0
        # We do this by running it through the forward pass and collecting mean stats
        ng.bench = False
        prev_layer = None
        for layer in layers:
            layer.fprop()
            if layer.weights is not None:
                mean = layer.get_activation_mean()
                scale = .5 #if prev_layer is None else prev_layer.reduction_factor()
                print "Scale weights: %.3f (%.3f) %s" % (scale/mean, scale, layer)
                layer.weights *= scale/mean
                layer.fprop()

            prev_layer = layer

        ng.bench = layer_bench

        start = drv.Event()
        end   = drv.Event()

        fprop_time  = 0
        bprop_time  = 0
        fprop_flops = 0
        bprop_flops = 0

        # We throw away the first run as it includes pycuda kernel loading times.
        # So add 1 to our loop count.
        for loop in range(loops+1):

            start.record()
            flops = 0

            #fprop
            for layer in layers:
                layer.fprop()
                flops += layer.flops
                if print_stats:
                    print "fprop:%8.3f mean %9.3f max %s" % (layer.get_activation_mean(), layer.get_activation_max(), layer)

            end.record()
            end.synchronize()
            msecs = end.time_since(start)
            print "fprop(%2d): %8.3f msecs %8.3f gflops" % (loop, msecs, flops / (msecs * 1000000.0))
            if loop > 0:
                fprop_time  += msecs
                fprop_flops += flops

            # HACK: omit softmax and cost layers to compare to Soumith numbers:
            last_layer = layers[-1]
            last_layer.bprop_in    = last_layer.fprop_out
            last_layer.bprop_in_ew = last_layer.fprop_out_ew

            start.record()
            flops = 0

            #bprop
            for layer in layers[:0:-1]:
                layer.bprop()
                layer.update(momentum, learning_rate)
                flops += layer.flops * 2
                #if type(layer) is PoolLayer:
                #set_trace()
                if print_stats:
                    print "bprop:%8.3f mean %9.3f max %s" % (layer.get_delta_mean(),  layer.get_delta_max(), layer)
                    if layer.weights is not None:
                        up_mean, up_max = (layer.get_update_mean(), layer.get_update_max())
                        wt_mean, wt_max = (layer.get_weight_mean(), layer.get_weight_max())
                        rt_mean, rt_max = (0.0001 * up_mean/wt_mean, 0.0001 * up_max/wt_max)
                        print "updat:%8.3f mean %9.3f max %s" % (up_mean, up_max, layer)
                        print "weigh:%8.3f mean %9.3f max" % (wt_mean, wt_max)
                        print "ratio:%8.3f mean %9.3f max" % (rt_mean, rt_max)

            end.record()
            end.synchronize()
            msecs = end.time_since(start)
            print "bprop(%2d): %8.3f msecs %8.3f gflops" % (loop, msecs, flops / (msecs * 1000000.0))
            if loop > 0:
                bprop_time  += msecs
                bprop_flops += flops

        if loops > 0:

            print "---------------------------------------------"
            print name, " Results:"
            print "---------------------------------------------"
            print "Avg(%d) fprop: %8.3f msecs %.3f gflops" % (
                loops, fprop_time/loops, fprop_flops / (fprop_time * 1000000.0))

            print "Avg(%d) bprop: %8.3f msecs %.3f gflops" % (
                loops, bprop_time/loops, bprop_flops / (bprop_time * 1000000.0))

            fprop_time  += bprop_time
            fprop_flops += bprop_flops

            print "Avg(%d) total: %8.3f msecs %.3f gflops\n\n" % (
                loops, fprop_time/loops, fprop_flops / (fprop_time * 1000000.0))
