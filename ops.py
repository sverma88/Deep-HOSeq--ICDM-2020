import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, inputFeatures, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, inputFeatures, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(seed=23))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv


def conv3d(input_, inputFeatures, output_dim,
           k_d=5, k_h=5, k_w=5, d_d=2, d_h=2, d_w=2, name="conv3d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_d, k_h, k_w, inputFeatures, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(seed=23))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))


        return conv


def conv3d_3x3(input_, inputFeatures, output_dim,
           k_d=5, k_h=5, k_w=5, d_d=2, d_h=2, d_w=2, name="conv3d_3x3"):
    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_d, k_h, k_w, inputFeatures, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(seed=23))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv


def conv2d_3x3(input_, inputFeatures, output_dim,
           k_d=3, k_h=3, d_d=2, d_h=2, name="conv2d_3x3"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_d, k_h, inputFeatures, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(seed=23))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_d, d_h, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))


        return conv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, bias_start=0.0):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.contrib.layers.xavier_initializer(seed=23))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias



def denseV(input_, output_size, scope=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "denseV"):
        matrix = tf.get_variable("Matrix", [output_size, shape[1]], tf.float32,
                                 tf.contrib.layers.xavier_initializer(seed=23),
                                 regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        return matrix

def denseW(input_, scope=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "denseW"):
        matrix = tf.get_variable("Matrix", [shape[1]], tf.float32,tf.contrib.layers.xavier_initializer(seed=23),
                                 regularizer = tf.contrib.layers.l2_regularizer(scale=0.01))

        bias = tf.get_variable("bias", [1], initializer=tf.constant_initializer(0.0))

        return bias + tf.reduce_sum(tf.multiply(matrix, input_), 1, keep_dims=True)


def masking(scores, sequence_lengths, score_mask_value=tf.constant(-np.inf)):
  score_mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(scores)[1])
  score_mask_values = score_mask_value * tf.ones_like(scores)
  return tf.where(score_mask, scores, score_mask_values)


def attention(inputs, att_dim, sequence_lengths, scope=None):
  assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

  with tf.variable_scope(scope or 'attention'):
    word_att_W = tf.get_variable(name='att_W', shape=[att_dim, 1])

    projection = tf.layers.dense(inputs, att_dim, tf.nn.tanh, name='projection')

    alpha = tf.matmul(tf.reshape(projection, shape=[-1, att_dim]), word_att_W)
    alpha = tf.reshape(alpha, shape=[-1, get_shape(inputs)[1]])
    alpha = tf.nn.softmax(alpha)

    outputs = tf.reduce_sum(inputs * tf.expand_dims(alpha, 2), axis=1)
    return outputs, alpha


def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]

  return dims
