from builtins import range
from collections import namedtuple
from datetime import datetime
import csv
import math
import time

import tensorflow.python.platform
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# TODO: why is batch size 64 going OOM?
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('forward_only', False,
                            """Only run the forward pass.""")
tf.app.flags.DEFINE_boolean('forward_backward_only', False,
                            """Only run the forward-forward pass.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                           """The data format for Convnet operations.
                           Can be either NHWC or NCHW.
                           """)
tf.app.flags.DEFINE_string('csv_file', '',
                           """File to output timing information to in csv
                           format. If not file is passed in, csv file will
                           not be cteated.
                           """)

parameters = []

conv_counter = 1
pool_counter = 1
affine_counter = 1

TimingEntry = namedtuple(
    'TimingEntry', ['info_string', 'timestamp', 'num_batches', 'mean', 'sd'])

def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
    global conv_counter
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        if FLAGS.data_format == 'NCHW':
          strides = [1, 1, dH, dW]
        else:
          strides = [1, dH, dW, 1]
        conv = tf.nn.conv2d(inpOp, kernel, strides, padding=padType,
                            data_format=FLAGS.data_format)
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases,
                                         data_format=FLAGS.data_format),
                          conv.get_shape())
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        return conv1

def _affine(inpOp, nIn, nOut):
    global affine_counter
    global parameters
    name = 'affine' + str(affine_counter)
    affine_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        affine1 = tf.nn.relu_layer(inpOp, kernel, biases, name=name)
        parameters += [kernel, biases]
        return affine1

def _mpool(inpOp, kH, kW, dH, dW):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    if FLAGS.data_format == 'NCHW':
      ksize = [1, 1, kH, kW]
      strides = [1, 1, dH, dW]
    else:
      ksize = [1, kH, kW, 1]
      strides = [1, dH, dW, 1]
    return tf.nn.max_pool(inpOp,
                          ksize=ksize,
                          strides=strides,
                          padding='VALID',
                          data_format=FLAGS.data_format,
                          name=name)

def loss(logits, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels], 1)
    onehot_labels = tf.sparse_to_dense(
        concated, tf.stack([batch_size, 1000]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=onehot_labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def inference(images):
    conv1 = _conv (images, 3, 64, 3, 3, 1, 1, 'SAME')
    pool1 = _mpool(conv1,  2, 2, 2, 2)
    conv2 = _conv (pool1,  64, 128, 3, 3, 1, 1, 'SAME')
    pool2 = _mpool(conv2,  2, 2, 2, 2)
    conv3 = _conv (pool2,  128, 256, 3, 3, 1, 1, 'SAME')
    conv4 = _conv (conv3,  256, 256, 3, 3, 1, 1, 'SAME')
    pool4 = _mpool(conv4,  2, 2, 2, 2)
    conv5 = _conv (pool4,  256, 512, 3, 3, 1, 1, 'SAME')
    conv6 = _conv (conv5,  512, 512, 3, 3, 1, 1, 'SAME')
    pool6 = _mpool(conv6,  2, 2, 2, 2)
    conv7 = _conv (pool6,  512, 512, 3, 3, 1, 1, 'SAME')
    conv8 = _conv (conv7,  512, 512, 3, 3, 1, 1, 'SAME')
    pool8 = _mpool(conv8,  2, 2, 2, 2)
    resh1 = tf.reshape(pool8, [-1, 512 * 7 * 7])
    affn1 = _affine(resh1, 512 * 7 * 7, 4096)
    affn2 = _affine(affn1, 4096, 4096)
    affn3 = _affine(affn2, 4096, 1000)

    return affn3


def time_tensorflow_run(session, target, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  if not isinstance(target, list):
    target = [target]
  target_op = tf.group(*target)
  for i in range(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target_op)
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))
  return TimingEntry(info_string, datetime.now(), FLAGS.num_batches, mn, sd)

def store_data_in_csv(timing_entries):
  with open(FLAGS.csv_file, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for timing_entry in timing_entries:
      writer.writerow(
          [timing_entry.info_string, timing_entry.timestamp,
           timing_entry.num_batches, timing_entry.mean, timing_entry.sd])

def run_benchmark():
  global parameters
  timing_entries = []
  with tf.Graph().as_default():
    # Generate some dummy images.
    image_size = 224
    if FLAGS.data_format == 'NCHW':
      image_shape = [FLAGS.batch_size, 3, image_size, image_size]
    else:
      image_shape = [FLAGS.batch_size, image_size, image_size, 3]
    images = tf.Variable(tf.ones(image_shape, dtype=tf.float32))

    labels = tf.Variable(tf.ones([FLAGS.batch_size],
                                 dtype=tf.int32))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    last_layer = inference(images)

    # Build an initialization operation.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session('')
    sess.run(init)

    run_forward = True
    run_forward_backward = True
    if FLAGS.forward_only and FLAGS.forward_backward_only:
      raise ValueError("Cannot specify --forward_only and "
                       "--forward_backward_only at the same time.")
    if FLAGS.forward_only:
      run_forward_backward = False
    elif FLAGS.forward_backward_only:
      run_forward = False

    if run_forward:
      # Run the forward benchmark.
      timing_entries.append(time_tensorflow_run(sess, last_layer, "Forward"))

    if run_forward_backward:
      # Add a simple objective so we can calculate the backward pass.
      objective = loss(last_layer, labels)
      # Compute the gradient with respect to all the parameters.
      grad = tf.gradients(objective, parameters)
      # Run the backward benchmark.
      timing_entries.append(time_tensorflow_run(sess, grad, "Forward-backward"))

  if FLAGS.csv_file:
    store_data_in_csv(timing_entries)


def main(_):
  run_benchmark()


if __name__ == '__main__':
  tf.app.run()
