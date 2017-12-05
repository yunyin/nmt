import tensorflow as tf

def get_optimizer(optimizer='sgd', learning_rate=None, momentum=None):
  if not learning_rate: learning_rate = tf.Variable(0.01, trainable = False)
  else: learning_rate  = tf.Variable(learning_rate, trainable = False)
  if optimizer == 'sgd':
    return tf.train.GradientDescentOptimizer(learning_rate), learning_rate
  elif optimizer == 'momentum':
    if not momentum: momentum = 0.9
    return tf.train.MomentumOptimizer(learning_rate, momentum), learning_rate
  elif optimizer == 'rmsprop':
    return tf.train.RMSPropOptimizer(learning_rate), learning_rate
  else:
    raise Exception("Invalid incoming optimizer name.")
