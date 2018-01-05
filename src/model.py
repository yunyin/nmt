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

  def ff_cell(self, in_tensor, in_size, out_size, dtype, scope):
    if len(in_tensor.get_shape()) != 2:
      tf.logging.info('in_tensor shape is not 2 for ff_cell')
      return None

    with tf.variable_scope(scope):
      W = tf.get_variable(
          "Weight", [in_size, out_size], dtype = dtype)
      b = tf.get_variable(
          "bias", [out_size], dtype = dtype)

    return tf.matmul(in_tensor, W) + b


  def init_decode_parameters(self, hidden_size, dtype):
    # variables for ~s(t)
    self.state_below_Wx = tf.get_variable(
        "state_below_Wx", [hidden_size, 2 * hidden_size], dtype = dtype)
    self.state_below_bx = tf.get_variable(
        "state_below_bx", [2 * hidden_size], dtype = dtype)

    self.state_below_W = tf.get_variable(
        "state_below_W", [hidden_size, hidden_size], dtype = dtype)
    self.state_below_b = tf.get_variable(
        "state_below_b", [hidden_size], dtype = dtype)

    self.gru_hs_W = tf.get_variable(
        "gru_hs_W", [hidden_size, 2 * hidden_size], dtype = dtype)
    self.gru_hs_Wb = tf.get_variable(
        "gru_hs_Wb", [2 * hidden_size], dtype = dtype)
    self.gru_hs_U = tf.get_variable(
        "gru_hs_U", [hidden_size, hidden_size], dtype = dtype)

    # variables for attention
    self.att_w = tf.get_variable(
        "att_w", [hidden_size, hidden_size], dtype = dtype)
    self.att_u = tf.get_variable(
        "att_u", [2 * hidden_size, hidden_size], dtype = dtype)
    self.att_v = tf.get_variable(
        "att_v", [hidden_size, 1], dtype = dtype)

    # variables for s(t)
    self.gru_s_W = tf.get_variable(
        "gru_s_W", [hidden_size, 2 * hidden_size], dtype = dtype)
    self.gru_s_Wb = tf.get_variable(
        "gru_s_Wb", [2 * hidden_size], dtype = dtype)
    self.gru_s_U = tf.get_variable(
        "gru_s_U", [hidden_size, hidden_size], dtype = dtype)
    self.contextWx = tf.get_variable(
        "contextWx", [2 * hidden_size, 2 * hidden_size], dtype = dtype)
    self.contextW = tf.get_variable(
        "contextW", [2 * hidden_size, hidden_size], dtype = dtype)

  def _build_encode(self, input, mask, vocab_size, hidden_size, dtype):
    with tf.name_scope('model/encode'):
      with tf.device("/cpu:0"):
        src_embedding = tf.get_variable(
            "src_embedding", [vocab_size, hidden_size], dtype = dtype)
        encode_inputs = tf.nn.embedding_lookup(src_embedding, input)
      encode_forward = self.lstm_cell(hidden_size)
      encode_backward = self.lstm_cell(hidden_size)
      batch_mask = tf.reduce_sum(mask, axis = 1)
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = encode_forward,
                                                               cell_bw = encode_backward,
                                                               inputs = encode_inputs,
                                                               sequence_length = tf.cast(batch_mask, tf.int32),
                                                               dtype = dtype)
    return tf.concat(outputs, axis = 2)

  def _build_decode(self, input, imask, encode, encode_mask, src_length, seq_length, vocab_size, hidden_size, dtype):
    with tf.name_scope('model/decode'):
      with tf.device("/cpu:0"):
        tgt_embedding = tf.get_variable(
            "tgt_embedding", [vocab_size, hidden_size], dtype = dtype)
        decode_inputs = tf.nn.embedding_lookup(tgt_embedding, input)

      self.init_states = tf.zeros([hidden_size], dtype = dtype)

      # only for training init
      batch_size = tf.shape(decode_inputs)[0]
      self.decode_state = tf.zeros([batch_size, hidden_size], dtype = dtype)

      decode_state = self.decode_state
      decode_states = []
      decode_context = []
      with tf.variable_scope("RNN"):
        self.init_decode_parameters(hidden_size = hidden_size, dtype = dtype)
        flattern_inputs = tf.reshape(decode_inputs, [-1, hidden_size])

        state_belowx = tf.matmul(flattern_inputs, self.state_below_Wx) + self.state_below_bx
        state_belowx = tf.reshape(state_belowx, [-1, seq_length, 2 * hidden_size])

        state_below_ = tf.matmul(flattern_inputs, self.state_below_W) + self.state_below_b
        state_below_ = tf.reshape(state_below_, [-1, seq_length, hidden_size])

        for time_step in range(seq_length):
          if time_step > 0: tf.get_variable_scope().reuse_variables()
          # calc ~s(t)
          preact1 = tf.matmul(decode_state, self.gru_hs_W) + self.gru_hs_Wb
          preact1 = tf.sigmoid(preact1 + state_belowx[:, time_step, :])

          r1 = tf.slice(preact1, [0, 0], [-1, hidden_size])
          u1 = tf.slice(preact1, [0, hidden_size], [-1, hidden_size])

          preactx1 = r1 * tf.matmul(decode_state, self.gru_hs_U)
          preactx1 = tf.tanh(preactx1 + state_below_[:, time_step, :])

          hidden_decode_state = u1 * decode_state + (1. - u1) * preactx1
          hidden_decode_state = imask[:, time_step, None] * hidden_decode_state + \
                                (1. - imask[:, time_step, None]) * decode_state

          # calc attention
          p_state = tf.matmul(hidden_decode_state, self.att_w)
          p_ctx = tf.matmul(tf.reshape(encode, [-1, 2 * hidden_size]), self.att_u)
          p_ctx = tf.reshape(p_ctx, [-1, src_length, hidden_size])

          p_state = tf.tanh(p_state[:, None, :] + p_ctx)

          att_state = tf.matmul(tf.reshape(p_state, [-1, hidden_size]), self.att_v)
          att_state = tf.reshape(att_state, [-1, src_length])

          self.alpha = tf.nn.softmax(tf.exp(att_state) * encode_mask)

          # C(t)
          context = tf.reduce_sum(encode * self.alpha[:, :, None], axis = 1)
          context = tf.reshape(context, [-1, 2 * hidden_size])

          # calc s(t)
          preact2 = tf.matmul(hidden_decode_state, self.gru_s_W) + self.gru_s_Wb
          preact2 = preact2 + tf.matmul(context, self.contextWx)
          preact2 = tf.sigmoid(preact2)

          r2 = tf.slice(preact2, [0, 0], [-1, hidden_size])
          u2 = tf.slice(preact2, [0, hidden_size], [-1, hidden_size])

          preactx2 = r2 * tf.matmul(hidden_decode_state, self.gru_s_U)
          preactx2_ = tf.matmul(context, self.contextW)
          preactx2 = tf.tanh(preactx2 + preactx2_)

          decode_state = u2 * hidden_decode_state + (1. - u2) * preactx2
          decode_state = imask[:, time_step, None] * decode_state + \
                         (1. - imask[:, time_step, None]) * hidden_decode_state

          decode_states.append(decode_state)
          decode_context.append(context)

    return decode_inputs, decode_states, decode_context

  def __init__(self, is_training, config, seq_length, optimizer = None, lr = None):
    hidden_size = config['hidden_size']
    self._optimizer = optimizer
    self._lr = lr

    with tf.name_scope('encode_input'):
      self.encode_input = tf.placeholder(tf.int32, [None, config['src_length']])
      self.encode_mask = tf.placeholder(tf.float32, [None, config['src_length']])

    self.encoder = self._build_encode(input = self.encode_input,
                                            mask = self.encode_mask,
                                            vocab_size = config['src_vocab_size'],
                                            hidden_size = config['hidden_size'],
                                            dtype = config['data_type'])

    with tf.name_scope('decode_input'):
      self.decode_input = tf.placeholder(tf.int32, [None, seq_length])
      self.decode_imask = tf.placeholder(tf.float32, [None, seq_length])

    decode_emb, self.decode_states, ctx = \
        self._build_decode(input = self.decode_input,
                           imask = self.decode_imask,
                           encode = self.encoder,
                           encode_mask = self.encode_mask,
                           src_length = config['src_length'],
                           seq_length = seq_length,
                           vocab_size = config['tgt_vocab_size'],
                           hidden_size = config['hidden_size'],
                           dtype = config['data_type'])

    with tf.name_scope('decode_output'):
      self.decode_output = tf.placeholder(tf.int32, [None, seq_length])
      self.decode_omask = tf.placeholder(tf.float32, [None, seq_length])

    with tf.name_scope('loss'):
      state_stack = tf.reshape(tf.stack(axis = 1, values = self.decode_states), [-1, hidden_size])
      logit_lstm = self.ff_cell(in_tensor = state_stack,
                                in_size = hidden_size, out_size = hidden_size,
                                dtype = config['data_type'], scope = 'ff_logit_lstm')
      emb_stack = tf.reshape(decode_emb, [-1, hidden_size])
      logit_emb = self.ff_cell(in_tensor = emb_stack,
                               in_size = hidden_size, out_size = hidden_size,
                               dtype = config['data_type'], scope = 'ff_logit_emb')

      ctx_stack = tf.reshape(tf.stack(axis = 1, values = ctx), [-1, 2 * hidden_size])
      logit_ctx = self.ff_cell(in_tensor = ctx_stack,
                               in_size = 2 * hidden_size, out_size = hidden_size,
                               dtype = config['data_type'], scope = 'ff_logit_ctx')

#      logits = tf.tanh(logit_lstm)
      logits = tf.tanh(logit_lstm + logit_emb + logit_ctx)
      logits = self.ff_cell(in_tensor = logits,
                            in_size = hidden_size, out_size = config['tgt_vocab_size'],
                            dtype = config['data_type'], scope = 'logit_calc')
      logits = tf.reshape(logits, [-1, seq_length, config['tgt_vocab_size']])

      self.probs = tf.nn.softmax(logits)

      loss = tf.contrib.seq2seq.sequence_loss(
          logits = logits,
          targets = self.decode_output,
          weights = self.decode_omask,
          average_across_timesteps = False,
          average_across_batch = False)

      self._cost = cost = tf.reduce_sum(loss)

    if not is_training: return

    with tf.name_scope('optimize'):
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                        config['max_grad_norm'])
      self._train_op = self._optimizer.apply_gradients(zip(grads, tvars),
                 global_step = tf.train.get_or_create_global_step())

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
