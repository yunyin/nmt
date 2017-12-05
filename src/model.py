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


  def __init__(self, is_training, config, optimizer = None, lr = None):
    self.config = config
    batch_size = config['batch_size']
    hidden_size = config['hidden_size']

    self._optimizer = optimizer
    self._lr = lr

    with tf.name_scope('inputs'):
      self.input_data = tf.placeholder(tf.int32, [batch_size, config['src_length']])
      self.input_mask = tf.placeholder(config['data_type'], [batch_size, config['src_length']])
      self.target_data = tf.placeholder(tf.int32, [batch_size, config['tgt_length']])
      self.target_mask = tf.placeholder(config['data_type'], [batch_size, config['tgt_length']])

    with tf.name_scope('model/encode'):
      with tf.device("/cpu:0"):
        src_embedding = tf.get_variable(
            "src_embedding", [config['src_vocab_size'], hidden_size], dtype = config['data_type'])
        inputs = tf.nn.embedding_lookup(src_embedding, self.input_data)
      encode_forward = self.lstm_cell(hidden_size)
      encode_backward = self.lstm_cell(hidden_size)
      batch_mask = tf.reduce_sum(self.input_mask, axis = 1)
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = encode_forward,
                                                               cell_bw = encode_backward,
                                                               inputs = inputs,
                                                               sequence_length = tf.cast(batch_mask, tf.int32),
                                                               dtype = config['data_type'])
      self._encoder = tf.concat(outputs, axis = 2)

    with tf.name_scope('model/decode'):
      with tf.device("/cpu:0"):
        tgt_embedding = tf.get_variable(
            "tgt_embedding", [config['tgt_vocab_size'], hidden_size], dtype = config['data_type'])
        targets = tf.nn.embedding_lookup(tgt_embedding, self.target_data)

      self._decode_initial_state = tf.zeros([batch_size, hidden_size], dtype = config['data_type'])

      decode_state = self._decode_initial_state
      outputs = []
      with tf.variable_scope("RNN"):
        self.init_decode_parameters(config = config)
        flattern_targets = tf.reshape(targets, [-1, hidden_size])
        state_belowx = tf.matmul(flattern_targets, self.state_below_Wx) + self.state_below_bx
        state_belowx = tf.reshape(state_belowx, [batch_size, config['tgt_length'], 2 * hidden_size])
        state_below_ = tf.matmul(flattern_targets, self.state_below_W) + self.state_below_b
        state_below_ = tf.reshape(state_below_, [batch_size, config['tgt_length'], hidden_size])

        self.target_mask_ = tf.reshape(self.target_mask, [batch_size, config['tgt_length'], 1])
        for time_step in range(config['tgt_length'] - 1):
          if time_step > 0: tf.get_variable_scope().reuse_variables()
          # calc ~s(t)
          preact1 = tf.matmul(decode_state, self.gru_hs_W) + self.gru_hs_Wb
          preact1 = tf.sigmoid(preact1 + state_belowx[:, time_step, :])

          r1 = tf.slice(preact1, [0, 0], [batch_size, hidden_size])
          u1 = tf.slice(preact1, [0, hidden_size], [batch_size, hidden_size])

          preactx1 = r1 * tf.matmul(decode_state, self.gru_hs_U)
          preactx1 = tf.tanh(preactx1 + state_below_[:, time_step, :])

          hidden_decode_state = u1 * decode_state + (1. - u1) * preactx1
          hidden_decode_state = tf.multiply(hidden_decode_state, self.target_mask_[:, time_step]) + \
                                tf.multiply(decode_state, (1. - self.target_mask_[:, time_step]))

          # calc attention
          p_state = tf.matmul(decode_state, self.att_w)
          flatten_encoder = tf.reshape(self._encoder, [-1, 2 * hidden_size])
          p_ctx = tf.matmul(flatten_encoder, self.att_u)
          p_ctx = tf.reshape(p_ctx, [batch_size, config['src_length'], hidden_size])
          p_state = tf.reshape(p_state, [batch_size, 1, hidden_size])
          p_state = tf.tanh(p_state + p_ctx)
          p_state = tf.reshape(p_state, [-1, hidden_size])
          att_state = tf.matmul(p_state, self.att_v)
          att_state = tf.reshape(att_state, [batch_size, config['src_length']])
          self.alpha = tf.nn.softmax(tf.exp(att_state) * self.input_mask)

          # C(t)
          self.alpha = tf.reshape(self.alpha, [batch_size, config['src_length'], 1])
          context = tf.reduce_sum(self._encoder * self.alpha, axis = 1)
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

          decode_state = u2 * decode_state + (1. - u2) * preactx2
          decode_state = self.target_mask_[:, time_step] * decode_state + \
                         (1. - self.target_mask_[:, time_step]) * hidden_decode_state

          outputs.append(decode_state)


    with tf.name_scope('loss'):
      output = tf.reshape(tf.stack(axis = 1, values = outputs), [-1, hidden_size])
      softmax_w = tf.get_variable(
          "softmax_w", [hidden_size, config['tgt_vocab_size']], dtype = config['data_type'])
      softmax_b = tf.get_variable(
          "softmax_b", [config['tgt_vocab_size']], dtype = config['data_type'])
      logits = tf.matmul(output, softmax_w) + softmax_b

      logits = tf.reshape(logits, [batch_size, config['tgt_length'] - 1, config['tgt_vocab_size']])

      self.probs = tf.nn.softmax(logits)

      loss = tf.contrib.seq2seq.sequence_loss(
          logits = logits,
          targets = self.target_data[:, 1:],
          weights = self.target_mask[:, 1:],
          average_across_timesteps=True,
          average_across_batch=True)

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
  def initial_state(self):
    return self._initial_state

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
