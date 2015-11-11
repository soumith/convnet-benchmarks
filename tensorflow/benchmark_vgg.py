from datetime import datetime
import math
import time

import tensorflow.python.platform
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# TODO: why is batch size 64 going OOM?
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")

parameters = []

conv_counter = 1
pool_counter = 1
affine_counter = 1

def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
    global conv_counter
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(inpOp, kernel, [1, dH, dW, 1], padding=padType)
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
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
    return tf.nn.max_pool(inpOp,
                          ksize=[1, kH, kW, 1],
                          strides=[1, dH, dW, 1],
                          padding='VALID',
                          name=name)

def loss(logits, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, 1000]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels,
                                                            name='xentropy')
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
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i > num_steps_burn_in:
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

def run_benchmark():
  global parameters
  with tf.Graph().as_default():
    # Generate some dummy images.
    image_size = 224
    images = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))

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

    # Run the forward benchmark.
    time_tensorflow_run(sess, last_layer, "Forward")

    # Add a simple objective so we can calculate the backward pass.
    objective = loss(last_layer, labels)
    # Compute the gradient with respect to all the parameters.
    grad = tf.gradients(objective, parameters)
    # Run the backward benchmark.
    time_tensorflow_run(sess, grad, "Forward-backward")


def main(_):
  run_benchmark()


if __name__ == '__main__':
  tf.app.run()
