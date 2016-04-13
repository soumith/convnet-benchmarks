import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import get_output, get_all_params
import time
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(
    description=' convnet benchmarks on imagenet')
parser.add_argument('--arch', '-a', default='alexnet',
                    help='Convnet architecture \
                    (alexnet, googlenet, vgg, overfeat)')
parser.add_argument('--batch_size', '-B', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--num_batches', '-n', type=int, default=100,
                    help='number of minibatches')

args = parser.parse_args()

if args.arch == 'alexnet':
    from alexnet import build_model, image_sz
elif args.arch == 'googlenet':
    from googlenet import build_model, image_sz
elif args.arch == 'vgg':
    from vgg import build_model, image_sz
elif args.arch == 'overfeat':
    from overfeat import build_model, image_sz
else:
    raise ValueError('Invalid architecture name')


def time_theano_run(func, fargs, info_string):
    num_batches = args.num_batches
    num_steps_burn_in = 10
    durations = []
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = func(*fargs)
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            durations.append(duration)
    durations = np.array(durations)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches,
           durations.mean(), durations.std()))


def main():
    batch_size = args.batch_size
    print('Building model...')
    layer, input_var = build_model(batch_size=batch_size)
    labels_var = T.ivector('labels')
    output = get_output(layer)
    loss = T.nnet.categorical_crossentropy(
        T.nnet.softmax(output), labels_var).mean(
        dtype=theano.config.floatX)
    gradient = T.grad(loss, get_all_params(layer))

    print('Compiling theano functions...')
    forward_func = theano.function([input_var], output)
    full_func = theano.function([input_var, labels_var], gradient)
    print('Functions are compiled')

    images = np.random.rand(batch_size, 3, image_sz, image_sz).astype(np.float32)
    labels = np.random.randint(0, 1000, size=batch_size).astype(np.int32)

    time_theano_run(forward_func, [images], 'Forward')
    time_theano_run(full_func, [images, labels], 'Forward-Backward')

if __name__ == '__main__':
    main()
