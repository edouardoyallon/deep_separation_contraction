from __future__ import division

import ipdb
import math
import tensorflow as tf
#from tensorflow.python import control_flow_ops
import numpy as np
import joblib

import model_utils as mu

FLAGS = tf.app.flags.FLAGS


def one_hot_embedding(label, n_classes):
  """
  One-hot embedding
  Args:
    label: int32 tensor [B]
    n_classes: int32, number of classes
  Return:
    embedding: tensor [B x n_classes]
  """
  embedding_params = np.eye(n_classes, dtype=np.float32)
  with tf.device('/cpu:0'):
    params = tf.constant(embedding_params)
    embedding = tf.gather(params, label)
  return embedding

def conv2d(x, n_in, n_out, k, s,p='SAME', bias=False, phase_train=True,scope='conv'):
  with tf.variable_scope(scope):
    kernel = tf.Variable(
      tf.truncated_normal([k, k, n_in, n_out],
        stddev=math.sqrt(8/(k*k*n_out))),
      name='weight')
    tf.add_to_collection('weights', kernel)
    conv = tf.nn.conv2d(x, kernel, [1,s,s,1], padding=p)
  return conv

def batch_norm(x, n_out, phase_train, scope='bn', affine=True):
  """
  Batch normalization on convolutional maps.
  Args:
    x: Tensor, 4D BHWD input maps
    n_out: integer, depth of input maps
    phase_train: boolean tf.Variable, true indicates training phase
    scope: string, variable scope
    affine: whether to affine-transform outputs
  Return:
    normed: batch-normalized maps
  """
  with tf.variable_scope(scope):
    beta = tf.constant(0.0, shape=[n_out])
    gamma = tf.constant(1.0, shape=[n_out])
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.99)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train,
      mean_var_with_update,
      lambda: (ema.average(batch_mean), ema.average(batch_var)))

    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, 
      beta, gamma, 1e-3, affine)
  return normed

def block(x, n_in, n_out, subsample,  phase_train, scope='res_block'):
  with tf.variable_scope(scope):
    if subsample:
      y = conv2d(x, n_in, n_out, 3, 2, 'SAME', False,phase_train, scope='conv_1')
    else:
      y = conv2d(x, n_in, n_out, 3, 1, 'SAME', False,phase_train, scope='conv_1')

    y = \
      batch_norm(y, n_out, phase_train, scope='bn_1')

    y = tf.mul(tf.sign(y),tf.sqrt(tf.abs(y)+1e-5) + 0.1)

    y = conv2d(y, n_out, n_out, 3, 1, 'SAME', False, phase_train, scope='conv_2')
    y = batch_norm(y, n_out, phase_train, scope='bn_2')

    y = tf.mul(tf.sign(y), tf.sqrt(tf.abs(y)+1e-5) + 0.1)
    return y

def group(x, n_in, n_out, n, first_subsample, phase_train, scope='group'):
  with tf.variable_scope(scope):
    y = block(x, n_in, n_out, first_subsample, phase_train, scope='block_1')
    y = tf.cond(phase_train, lambda: tf.nn.dropout(y, 0.6), lambda: y)#0.6 : 89.5
    for i in range(n - 1):
      y = block(y, n_out, n_out, False, phase_train, scope='block_%d' % (i + 2))
      y = tf.cond(phase_train, lambda: tf.nn.dropout(y, 0.6), lambda: y)
  return y

def net(x, n_layer_per_block, n_classes, phase_train,number_channel, scope='deep_net'):
  with tf.variable_scope(scope):
    n1=number_channel
    n2=number_channel
    n3=number_channel
    n4=number_channel

    y = conv2d(x, 3, n1, 3, 1, 'SAME',False, phase_train,scope='conv_init')
    y = batch_norm(y, n1, phase_train, scope='bn_init')
    y = tf.nn.relu(y, name='relu_init')

    y = group(y, n1, n2, n_layer_per_block, False, phase_train, scope='group_1')
    y = group(y, n2, n3, n_layer_per_block, True, phase_train, scope='group_2')
    y = group(y, n3, n4, n_layer_per_block, True, phase_train, scope='group_3')

    y = tf.nn.avg_pool(y, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID', name='avg_pool')
    y = tf.squeeze(y, squeeze_dims=[1, 2])

    w = tf.get_variable('DW', [n4, n_classes],initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    tf.add_to_collection('weights', w)
    bias = tf.get_variable('bias', [n_classes], initializer=tf.constant_initializer(0.0))
    y=tf.nn.xw_plus_b(y, w, bias)
  return y

def loss(logits, labels,n_class, scope='loss'):
  with tf.variable_scope(scope):
    # entropy loss
    targets = one_hot_embedding(labels, n_class)
    entropy_loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, targets),
      name='entropy_loss')
    tf.add_to_collection('losses', entropy_loss)
    # weight l2 decay loss
    weight_l2_losses = [tf.nn.l2_loss(o) for o in tf.get_collection('weights')]
    weight_decay_loss = tf.mul(FLAGS.weight_decay, tf.add_n(weight_l2_losses),
      name='weight_decay_loss')
    tf.add_to_collection('losses', weight_decay_loss)
  for var in tf.get_collection('losses'):
    tf.scalar_summary('losses/' + var.op.name, var)
  # total loss
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def accuracy(logits, gt_label, scope='accuracy'):
  with tf.variable_scope(scope):
    pred_label = tf.argmax(logits, 1)
    acc = 1.0 - tf.nn.zero_fraction(
      tf.cast(tf.equal(pred_label, gt_label), tf.int32))
  return acc

def train_op(loss, global_step, learning_rate):
  params = tf.trainable_variables()
  gradients = tf.gradients(loss, params, name='gradients')
  optim = tf.train.MomentumOptimizer(learning_rate, 0.9)
  update = optim.apply_gradients(zip(gradients, params))
  with tf.control_dependencies([update]):
    train_op = tf.no_op(name='train_op')
  return train_op


def cifar_input_stream(records_path):
  reader = tf.TFRecordReader()
  filename_queue = tf.train.string_input_producer([records_path], None)
  _, record_value = reader.read(filename_queue)
  features = tf.parse_single_example(record_value,
    {
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
    })
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image = tf.reshape(image, [32,32,3])
  image = tf.cast(image, tf.float32)
  label = tf.cast(features['label'], tf.int64)
  return image, label


def normalize_image(image):
  meanstd = joblib.load(FLAGS.mean_std_path)
  mean, std = meanstd['mean'], meanstd['std']
  normed_image = (image - mean) / std
  return normed_image

def random_distort_image(image):
  image = tf.image.resize_image_with_crop_or_pad(image, 36, 36)
  image = tf.random_crop(image, [32, 32, 3])
  image = tf.image.random_flip_left_right(image)
  return image


def make_train_batch(train_records_path, batch_size):
  with tf.variable_scope('train_batch'):
    with tf.device('/cpu:0'):
      train_image, train_label = cifar_input_stream(train_records_path)
      train_image = normalize_image(train_image)
      train_image = random_distort_image(train_image)
      train_image_batch, train_label_batch = tf.train.shuffle_batch(
        [train_image, train_label], batch_size=batch_size, num_threads=4,
        capacity=50000,
        min_after_dequeue=1000)
  return train_image_batch, train_label_batch




def make_validation_train_batch(train_records_path, batch_size):
  with tf.variable_scope('eval_train_batch'):
    with tf.device('/cpu:0'):
      train_image, train_label = cifar_input_stream(train_records_path)
      train_image = normalize_image(train_image)

      train_image_batch, train_label_batch = tf.train.batch(
        [train_image, train_label], batch_size=batch_size, num_threads=1,
        capacity=50000)
  return train_image_batch, train_label_batch




def make_validation_batch(test_records_path, batch_size):
  with tf.variable_scope('evaluate_batch'):
    with tf.device('/cpu:0'):
      test_image, test_label = cifar_input_stream(test_records_path)
      test_image = normalize_image(test_image)
      test_image_batch, test_label_batch = tf.train.batch(
        [test_image, test_label], batch_size=batch_size, num_threads=1,
        capacity=10000)
  return test_image_batch, test_label_batch
