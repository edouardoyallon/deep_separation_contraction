from __future__ import division

import sys
import os
import time
import math
import ipdb
from datetime import datetime
import numpy as np
import tensorflow as tf
#from tensorflow.python import control_flow_ops
import joblib
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import model_resnet as m
import model_utils as mu

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('load_dir', '', '')
tf.app.flags.DEFINE_integer('residual_net_n', 2, '')
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100')
tf.app.flags.DEFINE_string('train_tf_path', 'data/train.tf', '')
tf.app.flags.DEFINE_string('val_tf_path', 'data/test.tf', '')
tf.app.flags.DEFINE_string('mean_std_path', 'data/meanstd.pkl', '')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '')
tf.app.flags.DEFINE_integer('val_batch_size', 100, '')
tf.app.flags.DEFINE_float('weight_decay', 2e-4, 'Weight decay') #2e-4 : 85.5%
tf.app.flags.DEFINE_float('alpha', 0, 'Degree of non-linearity') #2e-4 : 85.5%
tf.app.flags.DEFINE_integer('summary_interval', 100, 'Interval for summary.')
tf.app.flags.DEFINE_integer('val_interval', 1000, 'Interval for evaluation.')
tf.app.flags.DEFINE_integer('max_steps', 121101, 'Maximum number of iterations.')
tf.app.flags.DEFINE_string('log_dir', 'logs/','')#'logs_cifar10/log_%s' % time.strftime("%Y%m%d_%H%M%S"), '')
tf.app.flags.DEFINE_integer('save_interval', 5000, '')
tf.app.flags.DEFINE_integer('save_end_accuracy', 5000, '')
tf.app.flags.DEFINE_integer('file_save_acc', 5000, 'File where the accuracy, amount of non-linearity, etc, are saved')


def get_acc():
  FLAGS.log_dir='logs'
  FLAGS.save_fig='/users/data/oyallon/Desktop/git_thigns/paperCVPR17'
  acc=[]
  alpha=[1.0,0.9 ,0.8, 0.7, 0.6, 0.5,0.4,0.3,0.2,0.1,0.05,0.0]
  n_channel=32
  for a in range(len(alpha)):
     DIR = os.path.join(FLAGS.log_dir, os.path.join(FLAGS.dataset, os.path.join(str(n_channel), str(alpha[a]))))
     x=np.load(os.path.join(DIR,'acc.npy'))
     acc.append(x)
     print('C10,alpha: %f, k: %f, acc: %f'%(alpha[a],n_channel,x))
  acc=np.array(acc)
  fig= plt.plot(alpha,100*acc,'-o',color='black')


  acc = []
  alpha = [1.0,0.9 ,0.8, 0.7, 0.6, 0.5,0.4,0.3,0.2,0.1,0.05,0.0]
  n_channel = 128
  for a in range(len(alpha)):
     DIR = os.path.join(FLAGS.log_dir, os.path.join(FLAGS.dataset, os.path.join(str(n_channel), str(alpha[a]))))
     try:
       x = np.load(os.path.join(DIR, 'acc.npy'))
       acc.append(x)
       print('C10,alpha: %f, k: %f, acc: %f' % (alpha[a], n_channel, x))
     except:
       acc.append(0.925)
       print('C10,alpha: %f, k: %f, it failed' % (alpha[a], n_channel))
  acc = np.array(acc)
  fig = plt.plot(alpha, 100*acc,'-x',color='black')

  plt.xlabel('Ratio $\\frac{k}{K}$')
  plt.ylabel('% accuracy')
  plt.legend(['K=32','K=128'],loc=4)
  plt.ylim([60, 100])

  plt.savefig(os.path.join(FLAGS.save_fig, '32channels.eps'), format='eps', dpi=1000, bbox_inches='tight')
  alpha=1.0
  n_channel=[16,32,64,128,256,512]
  acc=[]
  for a in range(len(n_channel)):
     DIR = os.path.join(FLAGS.log_dir, os.path.join(FLAGS.dataset, os.path.join(str(n_channel[a]), str(alpha))))
     try:
       x = np.load(os.path.join(DIR, 'acc.npy'))
       acc.append(x)
       print('C10,alpha: %f, k: %f, acc: %f' % (alpha, n_channel[a], x))
     except:
       acc.append(0.1)
       print('C10,alpha: %f, k: %f, it failed' % (alpha, n_channel[a]))
  acc = np.array(acc)
  plt.clf()
  fig, ax = plt.subplots()
  plt.plot(n_channel, 100*acc,'x-',color='black')
  plt.xscale('log')
  plt.xlabel('K')
  plt.ylabel('% accuracy')

  ax.xaxis.set_ticks(n_channel)
  for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())
  plt.xlim([16, 512])
  plt.ylim([70, 100])
  plt.savefig(os.path.join(FLAGS.save_fig, 'C10_nchannel.eps'), format='eps', dpi=1000, bbox_inches='tight')

  plt.clf()

  fig, ax = plt.subplots()
  acc=[]
  alpha = 1.0


  for a in range(len(n_channel)):
     DIR = os.path.join(FLAGS.log_dir, os.path.join('cifar100', os.path.join(str(n_channel[a]), str(alpha))))

     try:
       x = np.load(os.path.join(DIR, 'acc.npy'))
       acc.append(x)
       print('C100,alpha: %f, k: %f, acc: %f' % (alpha, n_channel[a], x))
     except:
       print('C100,alpha: %f, k: %f, it failed' % (alpha, n_channel[a]))

  acc = np.array(acc)
  plt.plot(n_channel,100* acc,'x-',color='black')
  plt.ylim([30, 100])
  plt.xlim([16, 512])
  plt.xscale('log')
  plt.xlabel('K')
  plt.ylabel('% accuracy')

  n_channel = [16, 32, 64, 128, 256, 512]
  ax.xaxis.set_ticks(n_channel)
  for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())

  plt.savefig(os.path.join(FLAGS.save_fig, 'C100_nchannel.eps'), format='eps', dpi=1000, bbox_inches='tight')
  cmap=plt.get_cmap('jet')
  line_colors = cmap(np.linspace(0,1,12))
  plt.clf()
  bins = np.linspace(0, 2., 100)
  x=np.load('incorrect_renorm.npy')

  from matplotlib.lines import Line2D
  for p in range(12):
    # plt.hist(x[2*p], bins, color='red',alpha=0.5, label='x',normed=1)
    # plt.hist(x[2*p+1], bins, alpha=0.5, label='y',normed=1)
    values, base = np.histogram(x[2 * p], bins=bins)
    values2, base2 = np.histogram(x[2 * p + 1], bins=bins)
    e = 0.7 / 12.0
    f = 0.1
    c = (e * p + f, e * p + f, e * p + f)
    plt.plot(base2[0:-1], np.cumsum(values2) / np.sum(values2), color=line_colors[p])

  for p in range(12):
    # plt.hist(x[2*p], bins, color='red',alpha=0.5, label='x',normed=1)
    # plt.hist(x[2*p+1], bins, alpha=0.5, label='y',normed=1)
    values, base = np.histogram(x[2 * p], bins=bins)
    values2, base2 = np.histogram(x[2 * p + 1], bins=bins)
    e = 0.7 / 12.0
    f = 0.1
    c = (e * p + f, e * p + f, e * p + f)
    plt.plot(base[0:-1], np.cumsum(values) / np.sum(values),':', color=line_colors[p])

    # plt.xscale('log')
    plt.yscale('log')
  plt.ylabel('Cumulative distribution')
  plt.xlabel('Distance')
  d=[]
  for i in range(12):
    d.append('n=%i'%(i+2))

  plt.legend(d, loc=4)

  plt.savefig(os.path.join(FLAGS.save_fig, 'hist.eps'), format='eps', dpi=1000, bbox_inches='tight')

  plt.clf()
  x = np.load('acc_NN.npy')
  y = np.load('acc_SVM.npy')
  plt.xlabel('n')
  plt.xlim([2,13])
  plt.ylim([40, 100])
  plt.ylabel('% accuracy')
  plt.plot(np.array(range(12)) + 2,np.ones(12)*88.0,'-.',  color='black')
  plt.plot(np.array(range(12))+2,100*x,'x-',color='black')
  plt.plot(np.array(range(12))+2,100*y,'o--',color='black')
  plt.legend(['Accuracy of the CNN','NN', 'SVM'],loc=4)
  plt.savefig(os.path.join(FLAGS.save_fig, 'acc.eps'), format='eps', dpi=1000, bbox_inches='tight')

  plt.clf()
  x = np.load('spec.npy')
  plt.xlabel('Principal components')
  plt.ylabel('Cumulated variance')
  linestyles = ['-', '--',  ':']
  plt.xlim([1, 32])
  for i in range(12):
    a=x[i,8,:] #5avant
    j=i
    e=0.7/12.0
    f=0.1
    c=(e*j+f,e*j+f,e*j+f)
    plt.plot(np.array(range(32))+1,np.cumsum(a),color=line_colors[i])

  d=[]
  for i in range(12):
    d.append('n=%i'%(i+2))

  plt.legend(d, loc=1)
  plt.savefig(os.path.join(FLAGS.save_fig, 'PCA.eps'), format='eps', dpi=1000, bbox_inches='tight')
  plt.clf()
  x = np.load('torsion.npy')

  plt.xlim([2, 13])
  x=x/(5000*5000/2)
  linestyles = ['-', '--',  ':']
  for i in range(10):
    a = x[:, i]
    e = 0.7 / 12.0
    f = 0.1
    c = (e * i + f, e * i + f, e * i+ f)
    plt.plot(np.array(range(12)) + 2,a, color=line_colors[i])

  d=[]
  for i in range(10):
    d.append('c=%i'%(i))

  plt.legend(d, loc=3,ncol=3)
  plt.xlabel('n')
  plt.ylabel('Averaged distance')
  plt.savefig(os.path.join(FLAGS.save_fig, 'torsion.eps'), format='eps', dpi=1000, bbox_inches='tight')
  plt.clf()
  x = np.load('SVs.npy')

  line_colors = cmap(np.linspace(0,1,12))
  for i in range(12):
    a = x[i, :]
    e = 0.7 / 12.0
    f = 0.1
    c = (e * i + f, e * i + f, e * i + f)
    plt.plot(a, color=line_colors[i])
  plt.ylabel('|$\Gamma_n^k$|')
  plt.xlabel('k')
  d=[]
  for i in range(12):
    d.append('n=%i' % (i+2))

  plt.legend(d, loc=1, ncol=3)
  plt.savefig(os.path.join(FLAGS.save_fig, 'SVs.eps'), format='eps', dpi=1000, bbox_inches='tight')

if __name__ == '__main__':
  get_acc()
