# -*- coding:utf-8 -*-
import os
import sys
import time

import inspect
import numpy as np
import tensorflow as tf

class Model():
  def lstm_cell(self, hidden_size):
    if 'reuse' in inspect.getargspec(
        tf.contrib.rnn.BasicLSTMCell.__init__).args:
      return tf.contrib.rnn.BasicLSTMCell(
          hidden_size, forget_bias = 0.0, state_is_tuple = True,
          reuse = tf.get_variable_scope().reuse)
    else:
      return tf.contrib.rnn.BasicLSTMCell(
          hidden_size, forget_bias = 0.0, state_is_tuple = True)

  def init_decode_parameters(self, config):
    # variables for ~s(t)
    self.state_below_Wx = tf.get_variable(
        "state_below_Wx", [config['hidden_size'], 2 * config['hidden_size']], dtype = config['data_type'])
    self.state_below_bx = tf.get_variable(
        "state_below_bx", [2 * config['hidden_size']], dtype = config['data_type'])

    self.state_below_W = tf.get_variable(
        "state_below_W", [config['hidden_size'], config['hidden_size']], dtype = config['data_type'])
    self.state_below_b = tf.get_variable(
        "state_below_b", [config['hidden_size']], dtype = config['data_type'])

    self.gru_hs_W = tf.get_variable(
        "gru_hs_W", [config['hidden_size'], 2 * config['hidden_size']], dtype = config['data_type'])
    self.gru_hs_Wb = tf.get_variable(
        "gru_hs_Wb", [2 * config['hidden_size']], dtype = config['data_type'])
    self.gru_hs_U = tf.get_variable(
        "gru_hs_U", [config['hidden_size'], config['hidden_size']], dtype = config['data_type'])

    # variables for attention
    self.att_w = tf.get_variable(
        "att_w", [config['hidden_size'], config['hidden_size']], dtype = config['data_type'])
    self.att_u = tf.get_variable(
        "att_u", [2 * config['hidden_size'], config['hidden_size']], dtype = config['data_type'])
    self.att_v = tf.get_variable(
        "att_v", [config['hidden_size'], 1], dtype = config['data_type'])

    # variables for s(t)
    self.gru_s_W = tf.get_variable(
        "gru_s_W", [config['hidden_size'], 2 * config['hidden_size']], dtype = config['data_type'])
    self.gru_s_Wb = tf.get_variable(
        "gru_s_Wb", [2 * config['hidden_size']], dtype = config['data_type'])
    self.gru_s_U = tf.get_variable(
        "gru_s_U", [config['hidden_size'], config['hidden_size']], dtype = config['data_type'])
    self.contextWx = tf.get_variable(
        "contextWx", [2 * config['hidden_size'], 2 * config['hidden_size']], dtype = config['data_type'])
    self.contextW = tf.get_variable(
        "contextW", [2 * config['hidden_size'], config['hidden_size']], dtype = config['data_type'])

  def _build_encode(self, input, mask, config):
    hidden_size = config['hidden_size']
    with tf.name_scope('model/encode'):
      with tf.device("/cpu:0"):
        src_embedding = tf.get_variable(
            "src_embedding", [config['src_vocab_size'], hidden_size], dtype = config['data_type'])
        encode_inputs = tf.nn.embedding_lookup(src_embedding, input)
      encode_forward = self.lstm_cell(hidden_size)
      encode_backward = self.lstm_cell(hidden_size)
      batch_mask = tf.reduce_sum(mask, axis = 1)
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = encode_forward,
                                                               cell_bw = encode_backward,
                                                               inputs = encode_inputs,
                                                               sequence_length = tf.cast(batch_mask, tf.int32),
                                                               dtype = config['data_type'])
    return tf.concat(outputs, axis = 2)

  def _build_decode(self, input, imask, encode, encode_mask, seq_length, config):
    hidden_size = config['hidden_size']
    with tf.name_scope('model/decode'):
      with tf.device("/cpu:0"):
        tgt_embedding = tf.get_variable(
            "tgt_embedding", [config['tgt_vocab_size'], hidden_size], dtype = config['data_type'])
        decode_inputs = tf.nn.embedding_lookup(tgt_embedding, input)

      batch_size = tf.shape(decode_inputs)[0]
      decode_state = tf.zeros([batch_size, hidden_size], dtype = config['data_type'])

      decode_states = []
      with tf.variable_scope("RNN"):
        self.init_decode_parameters(config = config)
        flattern_inputs = tf.reshape(decode_inputs, [-1, hidden_size])

        state_belowx = tf.matmul(flattern_inputs, self.state_below_Wx) + self.state_below_bx
        state_belowx = tf.reshape(state_belowx, [batch_size, seq_length, 2 * hidden_size])

        state_below_ = tf.matmul(flattern_inputs, self.state_below_W) + self.state_below_b
        state_below_ = tf.reshape(state_below_, [batch_size, seq_length, hidden_size])

#        imask_ = tf.reshape(imask, [batch_size, seq_length, 1])
        for time_step in range(seq_length):
          if time_step > 0: tf.get_variable_scope().reuse_variables()
          # calc ~s(t)
          preact1 = tf.matmul(decode_state, self.gru_hs_W) + self.gru_hs_Wb
          preact1 = tf.sigmoid(preact1 + state_belowx[:, time_step, :])

          r1 = tf.slice(preact1, [0, 0], [batch_size, hidden_size])
          u1 = tf.slice(preact1, [0, hidden_size], [batch_size, hidden_size])

          preactx1 = r1 * tf.matmul(decode_state, self.gru_hs_U)
          preactx1 = tf.tanh(preactx1 + state_below_[:, time_step, :])

          hidden_decode_state = u1 * decode_state + (1. - u1) * preactx1
          hidden_decode_state = imask[:, time_step, None] * hidden_decode_state + \
                                (1. - imask[:, time_step, None]) * decode_state

          # calc attention
          p_state = tf.matmul(hidden_decode_state, self.att_w)
          p_ctx = tf.matmul(tf.reshape(encode, [-1, 2 * hidden_size]), self.att_u)
          p_ctx = tf.reshape(p_ctx, [batch_size, config['src_length'], hidden_size])

          p_state = tf.tanh(p_state[:, None, :] + p_ctx)

          att_state = tf.matmul(tf.reshape(p_state, [-1, hidden_size]), self.att_v)
          att_state = tf.reshape(att_state, [batch_size, config['src_length']])

          self.alpha = tf.nn.softmax(tf.exp(att_state) * encode_mask)

          # C(t)
#          self.alpha = tf.reshape(self.alpha, [batch_size, config['src_length'], 1])
          context = tf.reduce_sum(encode * self.alpha[:, :, None], axis = 1)
          context = tf.reshape(context, [batch_size, 2 * hidden_size])

          # calc s(t)
          preact2 = tf.matmul(hidden_decode_state, self.gru_s_W) + self.gru_s_Wb
          preact2 = preact2 + tf.matmul(context, self.contextWx)
          preact2 = tf.sigmoid(preact2)

          r2 = tf.slice(preact2, [0, 0], [batch_size, hidden_size])
          u2 = tf.slice(preact2, [0, hidden_size], [batch_size, hidden_size])

          preactx2 = r2 * tf.matmul(hidden_decode_state, self.gru_s_U)
          preactx2_ = tf.matmul(context, self.contextW)
          preactx2 = tf.tanh(preactx2 + preactx2_)

          decode_state = u2 * hidden_decode_state + (1. - u2) * preactx2
          decode_state = imask[:, time_step, None] * decode_state + \
                         (1. - imask[:, time_step, None]) * hidden_decode_state

          decode_states.append(decode_state)
    return decode_states

  def __init__(self, is_training, config, seq_length, optimizer = None, lr = None):
    hidden_size = config['hidden_size']
    self._optimizer = optimizer
    self._lr = lr

    with tf.name_scope('encode_input'):
      self.encode_input = tf.placeholder(tf.int32, [config['batch_size'], config['src_length']])
      self.encode_mask = tf.placeholder(tf.float32, [config['batch_size'], config['src_length']])

    self._encoder = self._build_encode(input = self.encode_input,
                                       mask = self.encode_mask,
                                       config = config)

    with tf.name_scope('decode_input'):
      self.decode_input = tf.placeholder(tf.int32, [config['batch_size'], seq_length])
      self.decode_imask = tf.placeholder(tf.float32, [config['batch_size'], seq_length])

    self.decode_states = self._build_decode(input = self.decode_input, imask = self.decode_imask,
                                             encode = self._encoder, encode_mask = self.encode_mask,
                                             seq_length = seq_length, config = config)

    with tf.name_scope('decode_output'):
      self.decode_output = tf.placeholder(tf.int32, [config['batch_size'], seq_length])
      self.decode_omask = tf.placeholder(tf.float32, [config['batch_size'], seq_length])

    with tf.name_scope('loss'):
      outputs = tf.reshape(tf.stack(axis = 1, values = self.decode_states), [-1, hidden_size])
      softmax_w = tf.get_variable(
          "softmax_w", [hidden_size, config['tgt_vocab_size']], dtype = config['data_type'])
      softmax_b = tf.get_variable(
          "softmax_b", [config['tgt_vocab_size']], dtype = config['data_type'])
      logits = tf.matmul(outputs, softmax_w) + softmax_b

      logits = tf.reshape(logits, [config['batch_size'], seq_length, config['tgt_vocab_size']])

      self.probs = tf.nn.softmax(logits)

      loss = tf.contrib.seq2seq.sequence_loss(
          logits = logits,
          targets = self.decode_output,
          weights = self.decode_omask,
          average_across_timesteps=True,
          average_across_batch=False)

      self._cost = cost = tf.reduce_sum(loss)

    if not is_training: return

    with tf.name_scope('optimize'):
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                        config['max_grad_norm'])
      self._train_op = self._optimizer.apply_gradients(zip(grads, tvars),
                 global_step = tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(tf.float32, shape=[])
    self._lr_update = tf.assign(self._lr, self._new_lr)

  @property
  def train_op(self):
    return self._train_op

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def lr(self):
    return self._lr

  def assign_lr(self, sess, new_lr):
    sess.run(self._lr_update, feed_dict={self._new_lr: new_lr})
