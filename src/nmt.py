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
flags.DEFINE_string("gen_file", None, "File For Gen")
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

def run_epoch(sess, model, data, is_training = False,
              t_model = None, src_vocab = None, tgt_vocab = None):
  data.reset()
  costs = 0
  times = 0
  words = 0
  while data.has_next_batch():
    x_batch, y_batch, x_mask, y_mask = data.next_batch()

    feed_dict = {model.encode_input: x_batch,
                 model.encode_mask: x_mask,
                 model.decode_input: y_batch[:, :-1],
                 model.decode_imask: y_mask[:, :-1],
                 model.decode_output: y_batch[:, 1:],
                 model.decode_omask: y_mask[:, 1:]}
    fetches = {"probs": model.probs}
    if is_training: fetches["train_op"] = model.train_op

    vals = sess.run(fetches, feed_dict)

    real_index = y_batch[:, 1:].reshape(-1)
    real_index = [real_index[i] + i * np.shape(vals["probs"])[2] for i in range(len(real_index))]
    real_probs = vals["probs"].reshape(-1)[real_index]
    costs += -1.0 * np.sum(np.log(real_probs) * y_mask[:, 1:].reshape(-1))

    words += np.sum(y_mask[:, 1:])
    times += 1
    if times % 2000 == 100:
      tf.logging.info('step %d: training_loss: %.4f' %
                      (times, np.power(2, costs / words)))
    if times % 20000 == 0 and t_model != None:
      for idx in range(len(x_batch)):
        output, scores = sample_run(sess, t_model, x_batch[idx], x_mask[idx])
        in_words = [src_vocab.id2char(c) for c in x_batch[idx]]
        tf.logging.info('Input: %s' % (" ".join(in_words)))

        re_words = [tgt_vocab.id2char(c) for c in y_batch[idx]]
        tf.logging.info('Real: %s' % (" ".join(re_words)))

        out_words = [tgt_vocab.id2char(c) for c in output[0]]
        tf.logging.info('Output: %s' % (" ".join(out_words)))

  return costs / words

def sample_run(sess, model, src_in, src_mask, bos_id = 0, eos_id = 1, maxans = 5, maxlen = 50):
  sample = []
  sample_score = []
  # one sentence per sample_run
  src_in = np.reshape(src_in, [1, -1])
  src_mask = np.reshape(src_mask, [1, -1])
  ctx0 = sess.run(model.encoder,
                  {model.encode_input: src_in, model.encode_mask: src_mask})

  next_state = sess.run(model.init_states)
  next_state = np.tile(next_state, [1, 1])
  next_w = np.array([bos_id])

  live_k = 1
  dead_k = 0

  hyp_samples = [[]] * live_k
  hyp_scores = np.zeros(live_k).astype('float32')
  for i in range(maxlen):
    ctx = np.tile(ctx0, [live_k, 1, 1])
    next_w = np.reshape(next_w, [next_w.shape[0], 1])
    next_m = np.ones_like(next_w)
    fetches = {"next_state": model.decode_states,
               "probs": model.probs}
    feed_dict = {model.decode_input: next_w,
                 model.decode_imask: next_m,
                 model.decode_state: next_state,
                 model.encode_mask: src_mask,
                 model.encoder: ctx}
    vals = sess.run(fetches, feed_dict)

    probs = np.reshape(vals['probs'], [vals['probs'].shape[0], vals['probs'].shape[2]])
    next_state = vals['next_state'][-1]

    cand_scores = hyp_scores[:, None] - np.log(probs)
    vocab_size = probs.shape[1]
    cand_flat = cand_scores.flatten()

    this_count = maxans - dead_k
    ranks_flat = cand_flat.argsort()[:(this_count)]
    trans_idx = ranks_flat / vocab_size
    word_idx = ranks_flat % vocab_size

    new_hyp_samples = []
    new_hyp_scores = np.zeros(this_count).astype('float32')
    new_hyp_states = []
    for idx, [ti, wi] in enumerate(zip(trans_idx, word_idx)):
      new_hyp_samples.append(hyp_samples[ti] + [wi])
      new_hyp_scores[idx] = copy.copy(cand_scores[ti][wi])
      new_hyp_states.append(copy.copy(next_state[ti]))

    # check finished samples
    new_live_k = 0

    hyp_samples = []
    hyp_scores = []
    hyp_states = []
    for idx in range(len(new_hyp_samples)):
      if new_hyp_samples[idx][-1] == eos_id:
        sample.append(new_hyp_samples[idx])
        sample_score.append(new_hyp_scores[idx])
        dead_k += 1
      else:
        new_live_k += 1
        hyp_samples.append(new_hyp_samples[idx])
        hyp_scores.append(new_hyp_scores[idx])
        hyp_states.append(new_hyp_states[idx])
    hyp_scores = np.array(hyp_scores)
    live_k = new_live_k

    if live_k <= 0 or dead_k >= maxans: break
    next_w = np.array([w[-1] for w in hyp_samples])
    next_state = np.array(hyp_states)
  for idx in range(live_k):
    sample.append(hyp_samples[idx])
    sample_score.append(hyp_scores[idx])
  return sample, sample_score

def gen(config, gen_file):
  # load Vocab
  src_vocab = data_reader.Vocab(vocab_limits = config['src_vocab_size'])
  src_vocab.load_metadata(config['metadata']['src'])
  config['src_vocab_size'] = src_vocab.vocab_size()

  tgt_vocab = data_reader.Vocab(vocab_limits = config['tgt_vocab_size'])
  tgt_vocab.load_metadata(config['metadata']['tgt'])
  config['tgt_vocab_size'] = tgt_vocab.vocab_size()
  tf.logging.info(config)

  # create model
  with tf.name_scope('Genearte'):
    with tf.variable_scope("Model", reuse = None):
      gen_model = model.Model(is_training = False,
                              config = config,
                              seq_length = 1)

  sv = tf.train.Supervisor(logdir = config['logdir'])
  sess_config = tf.ConfigProto(allow_soft_placement = True,
                               log_device_placement = False)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5

  tf.logging.info('Start Sess')
  with sv.managed_session(config=sess_config) as sess:
    for line in open(gen_file):
      words = line.strip().split(' ')
      words.insert(0, "<s>")
      words.append("</s>")
      masks = [1] * len(words)
      if len(words) < config['src_length']:
        masks.extend([0] * (config['src_length'] - len(words)))
        words.extend(["</s>"] * (config['src_length'] - len(words)))
      src_vec = [src_vocab.char2id(c) for c in words]
      src_vec = np.array(src_vec)
      masks = np.array(masks)
      sample, scores = sample_run(sess, gen_model, src_vec, masks,
                                  bos_id = tgt_vocab.char2id("<s>"),
                                  eos_id = tgt_vocab.char2id("</s>"))
      out_words = [tgt_vocab.id2char(c) for c in sample[0]]
      tf.logging.info('Input: %s' % line.strip())
      tf.logging.info('Output: %s' % (" ".join(out_words)))

def train(config):
  # load Vocab
  src_vocab = data_reader.Vocab(vocab_limits = config['src_vocab_size'])
  src_vocab.load_metadata(config['metadata']['src'])
  config['src_vocab_size'] = src_vocab.vocab_size()

  tgt_vocab = data_reader.Vocab(vocab_limits = config['tgt_vocab_size'])
  tgt_vocab.load_metadata(config['metadata']['tgt'])
  config['tgt_vocab_size'] = tgt_vocab.vocab_size()
  tf.logging.info(config)

  initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])

  # create models
  with tf.name_scope('Train'):
    opt, lr = optimizer.get_optimizer(config['optimizer'], config['learning_rate'])
    with tf.variable_scope("Model", reuse = None, initializer = initializer):
      train_model = model.Model(is_training = True,
                                config = config,
                                seq_length = config['tgt_length'] - 1,
                                optimizer = opt,
                                lr = lr)

  with tf.name_scope('Test'):
    with tf.variable_scope("Model", reuse = True):
      test_model = model.Model(is_training = False,
                               config = config,
                               seq_length = 1)

  sv = tf.train.Supervisor(logdir = config['logdir'])
  sess_config = tf.ConfigProto(allow_soft_placement = True,
                               log_device_placement = True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9

  # load Data
  train_data = data_reader.DataReader(src_data = config['train_data']['src'][0],
                                      tgt_data = config['train_data']['tgt'][0],
                                      src_vocab = src_vocab,
                                      tgt_vocab = tgt_vocab,
                                      src_length = config['src_length'],
                                      tgt_length = config['tgt_length'],
                                      batch_size = config['batch_size'])

  tf.logging.info('Start Sess')
  with sv.managed_session(config=sess_config) as sess:
    for i in range(config['n_epoch']):
      lr_decay = config['lr_decay'] ** max(i + 1 - config['decay_epoch'], 0)
      train_model.assign_lr(sess, config['learning_rate'] * lr_decay)

      tf.logging.info('Iter %d Start, Learning_rate: %.4f' % (i, sess.run(train_model.lr)))
      loss = run_epoch(sess, train_model, train_data, is_training = True, \
                       t_model = test_model, src_vocab = src_vocab, tgt_vocab = tgt_vocab)
      tf.logging.info('Iter %d: training_loss: %.4f' % (i, np.power(2, loss)))

def main(_):
  config = HParam()
  if not FLAGS.type:
    raise ValueError('run type is train')
  if FLAGS.type == 'train':
    train(config)
  elif FLAGS.type == 'gen':
    gen(config, FLAGS.gen_file)
  else:
    raise ValueError('only train is valid')

if __name__ == '__main__':
  tf.app.run()
