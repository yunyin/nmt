# -*- coding:utf-8 -*-
import os
import sys
import time
import numpy as np

class Vocab():
  def __init__(self, vocab_limits = -1):
    self._vocab_size = vocab_limits

  def create_vocab(self, datafiles, metadata, vocab_limits = -1):
    print 'Start Vocab Create'
    sys.stdout.flush()
    total_data = []
    for line in open(datafiles):
      words = line.strip().split(' ')
      #words = list(line.strip().replace(" ", "").decode('utf-8'))
      total_data.extend(words)

    print 'Data Load End For Vocab Create'
    sys.stdout.flush()
    words = list(set(total_data))
    words.sort()
    words.insert(0, '<unk>')
    words.insert(0, '</s>')
    words.insert(0, '<s>')

    if vocab_limits == -1: self._vocab_size = len(words)
    else: self._vocab_size = min(vocab_limits, len(words))
    words = words[:self._vocab_size]

    print 'Vocabulary Size: %d' % self._vocab_size
    sys.stdout.flush()
    self.char2id_dict = {w: i for i, w in enumerate(words)}
    self.id2char_dict = {i: w for i, w in enumerate(words)}

    # save metadata file
    self.save_metadata(metadata)

  def vocab_size(self):
    return self._vocab_size

  def char2id(self, c):
    if not self.char2id_dict.has_key(c):
      c = '<unk>'
    return self.char2id_dict[c]

  def id2char(self, id):
    return self.id2char_dict[id]

  def load_metadata(self, file):
    if not os.path.exists(file):
      print 'Vocab Metadata {} does not exists'.format(file)
      sys.exit(-1)
    self.char2id_dict = dict()
    self.id2char_dict = dict()

    cnt = 0
    for line in open(file):
      idx, word = line.strip().decode('utf-8').split('\t')
      self.char2id_dict[word] = int(idx)
      self.id2char_dict[int(idx)] = word
      cnt += 1
      if cnt == self._vocab_size: break
    self._vocab_size = len(self.id2char_dict)
    print 'Loading Vocabulary Size:{}'.format(self._vocab_size)
    sys.stdout.flush()

  def save_metadata(self, file):
    with open(file, 'w') as f:
      for i in range(self._vocab_size):
        c = self.id2char(i)
        #f.write('{}\t{}\n'.format(i, c.encode('utf-8')))
        f.write('{}\t{}\n'.format(i, c))

class DataReader():
  def __init__(self,
               src_data, tgt_data,
               src_vocab, tgt_vocab,
               src_length = 1, tgt_length = 1,
               batch_size = 1):
    self.src_length = src_length
    self.tgt_length = tgt_length
    self.src_vocab = src_vocab
    self.tgt_vocab = tgt_vocab
    self.batch_size = batch_size

    print 'Start Read Data'
    sys.stdout.flush()
    self.src_data = []
    self.src_mask = []
    for line in open(src_data):
      words = line.strip().split(' ')
      words.insert(0, "<s>")
      words.append("</s>")
      masks = [1] * len(words)
      if len(words) < self.src_length:
        masks.extend([0] * (self.src_length - len(words)))
        words.extend(["</s>"] * (self.src_length - len(words)))
      words = words[:self.src_length]
      masks = masks[:self.src_length]
      assert len(masks) == self.src_length
      words = [self.src_vocab.char2id(c) for c in words]
      self.src_data.append(words)
      self.src_mask.append(masks)

    self.tgt_data = []
    self.tgt_mask = []
    for line in open(tgt_data):
      words = line.strip().split(' ')
      words.insert(0, "<s>")
      words.append("</s>")
      masks = [1] * len(words)
      if len(words) < self.tgt_length:
        masks.extend([0] * (self.tgt_length - len(words)))
        words.extend(["</s>"] * (self.tgt_length - len(words)))
      words = words[:self.tgt_length]
      masks = masks[:self.tgt_length]
      words = [self.tgt_vocab.char2id(c) for c in words]
      self.tgt_data.append(words)
      self.tgt_mask.append(masks)

    print 'Read Data End'
    sys.stdout.flush()

    assert len(self.src_data) == len(self.tgt_data)

    # pointer position to generate current batch
    self.reset()

  def reset(self):
    self.pointer = 0

  def has_next_batch(self):
    if self.pointer < len(self.src_data): return True
    return False

  def next_batch(self):
    x_batches = []
    y_batches = []
    x_masks = []
    y_masks = []
    for i in range(self.batch_size):
      if self.pointer >= len(self.src_data):
        bx = [0] * self.src_length
        by = [0] * self.tgt_length
        mx = [0] * self.src_length
        my = [0] * self.tgt_length
      else:
        bx = self.src_data[self.pointer]
        by = self.tgt_data[self.pointer]
        mx = self.src_mask[self.pointer]
        my = self.tgt_mask[self.pointer]

      x_batches.append(bx)
      y_batches.append(by)
      x_masks.append(mx)
      y_masks.append(my)
      self.pointer += 1

    return x_batches, y_batches, x_masks, y_masks

if __name__=='__main__':
  vocab = Vocab()
  vocab.create_vocab(datafiles = sys.argv[1],
                     metadata = sys.argv[2],
                     vocab_limits = int(sys.argv[3]))
  vocab.load_metadata(sys.argv[2])
