#####
## MODIFIED BY: Edouard Oyallon
## Team DATA - ENS 2016
## Can be found on: https://github.com/bgshih/tf_resnet_cifar
#####


from __future__ import division


import os
from datetime import datetime
import tensorflow as tf
import numpy as np


import model_cifar_contract as m_c
import model_cifar as m

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('load_dir', '', '')
tf.app.flags.DEFINE_integer('n_block', 2, '')
tf.app.flags.DEFINE_integer('n_channel', 128, '')
tf.app.flags.DEFINE_boolean('non_contract',False,'Allow to use the model with a non contractive non-linearity')
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100')
tf.app.flags.DEFINE_string('train_tf_path', 'data/train.tf', '')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '')
tf.app.flags.DEFINE_integer('val_batch_size', 100, '')
tf.app.flags.DEFINE_float('weight_decay', 2e-4, 'Weight decay') #2e-4 : 85.5%
tf.app.flags.DEFINE_float('alpha', 1, 'Degree of non-linearity') #2e-4 : 85.5%
tf.app.flags.DEFINE_integer('summary_interval', 100, 'Interval for summary.')
tf.app.flags.DEFINE_integer('val_interval', 1000, 'Interval for evaluation.')
tf.app.flags.DEFINE_integer('max_steps', 121101, 'Maximum number of iterations.')
tf.app.flags.DEFINE_string('log_dir', 'logs/','')#'logs_cifar10/log_%s' % time.strftime("%Y%m%d_%H%M%S"), '')
tf.app.flags.DEFINE_integer('save_interval', 5000, '')
tf.app.flags.DEFINE_integer('save_end_accuracy', 5000, '')


def train_and_val():
  with tf.Graph().as_default():
    FLAGS.train_tf_path = 'data_'+FLAGS.dataset+'/train.tf'
    FLAGS.val_tf_path = 'data_' + FLAGS.dataset + '/test.tf'
    FLAGS.mean_std_path = 'data_' + FLAGS.dataset + '/meanstd.pkl'
    FLAGS.log_dir = os.path.join(FLAGS.log_dir,os.path.join(FLAGS.dataset,os.path.join(str(FLAGS.n_channel),str(FLAGS.alpha))))
    n_class=10
    # model outputs
    if(FLAGS.dataset=='cifar10'):
      n_class=10
    elif (FLAGS.dataset == 'cifar100'):
      n_class=100

    if(FLAGS.non_contract):
      FLAGS.log_dir = os.path.join(FLAGS.log_dir,'non_contractive')

    # train/test phase indicator
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # learning rate is manually set
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    tf.scalar_summary('learning_rate', learning_rate)

    # global step
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # train/test inputs
    train_image_batch, train_label_batch = m.make_train_batch(
      FLAGS.train_tf_path, FLAGS.train_batch_size)
    val_image_batch, val_label_batch = m.make_validation_batch(
      FLAGS.val_tf_path, FLAGS.val_batch_size)
    image_batch, label_batch = tf.cond(phase_train,
                                                     lambda: (train_image_batch, train_label_batch),
                                                     lambda: (val_image_batch, val_label_batch))

    logits=[]
    if (not FLAGS.non_contract):
      print('We use alpha = %f, n_channel = %f, on dataset %s ' % (FLAGS.alpha,FLAGS.n_channel,FLAGS.dataset))
      logits = m.net(image_batch, FLAGS.n_block, n_class, phase_train, FLAGS.alpha,FLAGS.n_channel)
    else:
      logits = m_c.net(image_batch, FLAGS.n_block, n_class, phase_train,  FLAGS.n_channel)
    # total loss
    loss = m.loss(logits, label_batch,n_class)
    accuracy = m.accuracy(logits, label_batch)
    tf.scalar_summary('train_loss', loss)
    tf.scalar_summary('train_accuracy', accuracy)

    # train one step
    train_op = m.train_op(loss, global_step, learning_rate)

    # saver
    saver = tf.train.Saver(tf.all_variables())


    # start session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options))

    # summary writer
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(
      FLAGS.log_dir, graph=sess.graph)

    # initialize parameters or load from a checkpoint
    if FLAGS.load_dir != '':
      # load from checkpoint
      checkpoint = tf.train.get_checkpoint_state(FLAGS.load_dir)
      print(checkpoint)
      print(FLAGS.load_dir)
      model_checkpoint_path = checkpoint.model_checkpoint_path
      if checkpoint and model_checkpoint_path:
        saver.restore(sess, model_checkpoint_path)
        print('Model restored from %s' % model_checkpoint_path)
      else:
        raise 'Load directory provided by no checkpoint found'
    else:
      init_op = tf.initialize_all_variables()
      print('Initializing...')
      sess.run(init_op, {phase_train.name: True})

    print('Start training...')
    # train loop
    tf.train.start_queue_runners(sess=sess)
    curr_lr = 0.0
    for step in range(FLAGS.max_steps):
      _lr = np.power(0.5,2+np.floor(step/10000))

      if curr_lr != _lr:
        curr_lr = _lr
        print('Learning rate set to %f' % curr_lr)

      # train
      fetches = [train_op, loss]
      if step > 0 and step % FLAGS.summary_interval == 0:
        fetches += [accuracy, summary_op]
      sess_outputs = sess.run(
        fetches, {phase_train.name: True, learning_rate.name: curr_lr})

      # summary
      if step > 0 and step % FLAGS.summary_interval == 0:
        train_loss_value, train_acc_value, summary_str = sess_outputs[1:]
        print('[%s] Iteration %d, train loss = %f, train accuracy = %f' %
              (datetime.now(), step, train_loss_value, train_acc_value))
        summary_writer.add_summary(summary_str, step)

      # accuracy on the testset
      if step > 0 and step % FLAGS.val_interval == 0:
        print('Evaluating...')
        n_val_samples = 10000
        val_batch_size = FLAGS.val_batch_size
        n_val_batch = int(n_val_samples / val_batch_size)
        val_logits = np.zeros((n_val_samples, n_class), dtype=np.float32)
        val_labels = np.zeros((n_val_samples), dtype=np.int64)
        val_losses = []
        for i in range(n_val_batch):
          fetches = [logits, label_batch, loss]
          session_outputs = sess.run(
            fetches, {phase_train.name: False})
          val_logits[i*val_batch_size:(i+1)*val_batch_size, :] = session_outputs[0]
          val_labels[i*val_batch_size:(i+1)*val_batch_size] = session_outputs[1]
          val_losses.append(session_outputs[2])
        pred_labels = np.argmax(val_logits, axis=1)
        val_accuracy = np.count_nonzero(
          pred_labels == val_labels) / n_val_samples
        val_loss = float(np.mean(np.asarray(val_losses)))
        print('Test accuracy = %f' % val_accuracy)
        val_summary = tf.Summary()
        val_summary.value.add(tag='val_accuracy', simple_value=val_accuracy)
        val_summary.value.add(tag='val_loss', simple_value=val_loss)
        summary_writer.add_summary(val_summary, step)
        np.save(os.path.join(FLAGS.log_dir, 'acc.npy'),val_accuracy)

      # save variables
      if step % FLAGS.save_interval == 0 and step > 0:
        checkpoint_path = os.path.join(FLAGS.log_dir, 'checkpoint')
        saver.save(sess, checkpoint_path, global_step=step)
        print('Checkpoint saved at %s' % checkpoint_path)

if __name__ == '__main__':
  train_and_val()
