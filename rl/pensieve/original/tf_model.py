import tensorflow as tf

# load the TF checkpoint model
NN_MODEL = './pretrain_linear_reward.ckpt'

with tf.Session() as sess:
  sess.run(tf.compat.v1.global_variables_initializer())
  saver = tf.compat.v1.train.Saver() 

  # restore neural net parameters
  nn_model = NN_MODEL
  if nn_model is not None: 
      saver.restore(sess, nn_model)
