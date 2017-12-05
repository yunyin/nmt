# -*- coding:utf-8 -*-
import os
import sys
import time

import numpy as np
import tensorflow as tf
import json
import copy

import data_reader
import model
import optimizer

flags = tf.flags
flags.DEFINE_string("config", None, "Directory of Config File.")
flags.DEFINE_string("type", None, "Run Type: train/gen")
flags.DEFINE_integer("task_id", 0, "Task Id for Each workers")
flags.DEFINE_string("job_name", 'worker', "Job Type: ps/worker")
FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

def HParam():
  if not FLAGS.config:
    raise ValueError("Must set --config")

  if not os.path.exists(FLAGS.config):
    raise ValueError("Config File %s Does Not Exist" % (FLAGS.config))

  config = json.load(open(FLAGS.config))
  if config['data_type'] == 'float16':
    config['data_type'] = tf.float16
  else: config['data_type'] = tf.float32
  return config

def run_epoch(sess, model, data, is_training = False):
  data.reset()
  costs = 0
  iters = 0
  times = 0
  while data.has_next_batch():
    x_batch, y_batch, x_mask, y_mask = data.next_batch()
    feed_dict = {model.input_data: x_batch,
                 model.target_data: y_batch,
                 model.input_mask: x_mask,
                 model.target_mask: y_mask}
    fetches = {"costs": model._cost}
    if is_training: fetches["train_op"] = model.train_op

    vals = sess.run(fetches, feed_dict)
    costs += vals["costs"]

    iters += 1
    times += 1
    if times % 2000 == 100:
      tf.logging.info('step %d: training_loss: %.4f' % (times, np.exp(costs / iters)))

  return costs / iters

def train(config):
  # load Vocab
  src_vocab = data_reader.Vocab(vocab_limits = config['src_vocab_size'])
  src_vocab.load_metadata(config['metadata']['src'])
  config['src_vocab_size'] = src_vocab.vocab_size()

  tgt_vocab = data_reader.Vocab(vocab_limits = config['tgt_vocab_size'])
  tgt_vocab.load_metadata(config['metadata']['tgt'])
  config['tgt_vocab_size'] = tgt_vocab.vocab_size()
  tf.logging.info(config)

  # load Data
  train_data = data_reader.DataReader(src_data = config['train_data']['src'][0],
                                      tgt_data = config['train_data']['tgt'][0],
                                      src_vocab = src_vocab,
                                      tgt_vocab = tgt_vocab,
                                      src_length = config['src_length'],
                                      tgt_length = config['tgt_length'],
                                      batch_size = config['batch_size'])

  initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])

  # create models
  with tf.name_scope('Train'):
    opt, lr = optimizer.get_optimizer("sgd", config['learning_rate'])
    with tf.variable_scope("Model", reuse = None, initializer = initializer):
      train_model = model.Model(is_training = True,
                                config = config,
                                optimizer = opt,
                                lr = lr)

  sv = tf.train.Supervisor(logdir = config['logdir'])
  sess_config = tf.ConfigProto(allow_soft_placement = True,
                               log_device_placement = False)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5

  tf.logging.info('Start Sess')
  with sv.managed_session(config=sess_config) as sess:
    for i in range(config['n_epoch']):
      lr_decay = config['lr_decay'] ** max(i + 1 - config['decay_epoch'], 0)
      train_model.assign_lr(sess, config['learning_rate'] * lr_decay)

      tf.logging.info('Iter %d Start, Learning_rate: %.4f' % (i, sess.run(train_model.lr)))
      loss = run_epoch(sess, train_model, train_data, is_training = True)
      tf.logging.info('Iter %d: training_loss: %.4f' % (i, np.exp(loss)))

def main(_):
  config = HParam()
  if not FLAGS.type:
    raise ValueError('run type is train')
  if FLAGS.type == 'train':
    train(config)
  else:
    raise ValueError('only train is valid')

if __name__ == '__main__':
  tf.app.run()
