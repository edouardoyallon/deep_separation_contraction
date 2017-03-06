#####
## MODIFIED BY: Edouard Oyallon
## Team DATA - ENS 2016
## Can be found on: https://github.com/bgshih/tf_resnet_cifar
#####


from __future__ import division


import os
import tensorflow as tf
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from sklearn import preprocessing

import model_cifar as m

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

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
tf.app.flags.DEFINE_bool('recompute', False, '')




def build_statistics():
  n_train_samples = 50000
  n_val_samples = 10000
  FLAGS.train_tf_path = 'data_' + FLAGS.dataset + '/train.tf'
  FLAGS.val_tf_path = 'data_' + FLAGS.dataset + '/test.tf'
  FLAGS.mean_std_path = 'data_' + FLAGS.dataset + '/meanstd.pkl'
  FLAGS.log_dir = os.path.join(FLAGS.log_dir,
                               os.path.join(FLAGS.dataset, os.path.join(str(FLAGS.n_channel), str(FLAGS.alpha))))
  n_class = 10
  # model outputs
  if (FLAGS.dataset == 'cifar10'):
    n_class = 10
  elif (FLAGS.dataset == 'cifar100'):
    n_class = 100

  cando = False
  n_train_samples = 50000
  n_val_samples = 10000
  idx_cv = []
  SV_prev = []
  layer = 0

  N_SV = np.zeros((12, 30), dtype=float)

  acc_SVM = np.zeros((12))
  acc_NN = np.zeros((12))

  hist=[]
  hist_bad_2NN = np.zeros((12, 100))
  lipstchitz = np.zeros((12, 100))
  idx_couples_contraction = []
  distance=[]
  d=[]
  L=[]
  torsion=np.zeros((12,10))
  PCA=np.zeros((12,10,32))
  for group in range(3):
      for block in range(2):
          for conv in range(2):
              # First we reproduce the graphs used in our models
              with tf.Graph().as_default() as g:
                  phase_train = tf.constant(False,dtype=tf.bool)#tf.placeholder(tf.bool, name='phase_train')

                  training_set = tf.placeholder(tf.bool, name='training_set')

                  train_image_batch, train_label_batch = \
                    m.make_validation_train_batch(FLAGS.train_tf_path, FLAGS.val_batch_size)

                  val_image_batch, val_label_batch = m.make_validation_batch(
                  FLAGS.val_tf_path, FLAGS.val_batch_size)

                  image_batch, label_batch = tf.cond(training_set,
                                                 lambda: (train_image_batch, train_label_batch),
                                                 lambda: (val_image_batch, val_label_batch))

              # model outputs
                  logits = m.net(image_batch, FLAGS.n_block, n_class, phase_train, FLAGS.alpha, FLAGS.n_channel)


              # saver


                  name = 'deep_net/group_' + str(group+1) +'/block_'+str(block+1)+'/nonlinearity_'+str(conv+1)+'/concat'#'deep_net/xw_plus_b'#
                  print(name)
                  output=[v for v in g.get_operations() if (v.name == name)][0].outputs[0]
                  print(output)
                  s = output.get_shape()
                  print(s)
                  output_avg = tf.nn.avg_pool(output, [1, s[1], s[2], 1], [1, 1, 1, 1], 'VALID')  # ,name=name+'avg')
                  output_avg = tf.squeeze(output_avg, squeeze_dims=[1, 2])  # ,name=name+'sqz')
                  #output_avg=output
                  print(output_avg)
                  # start session
                  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
                  #tf.logging.set_verbosity(tf.logging.ERROR)


                  saver = tf.train.Saver(tf.all_variables())

                  sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))  # , gpu_options=gpu_options))

                  init_op = tf.initialize_all_variables()
                  sess.run(init_op)

                  # load the models
                  checkpoint = tf.train.get_checkpoint_state(FLAGS.log_dir)
                  model_checkpoint_path = checkpoint.model_checkpoint_path
                  if checkpoint and model_checkpoint_path:
                      saver.restore(sess, model_checkpoint_path)
                      print('Model restored from %s' % model_checkpoint_path)
                  else:
                      raise 'Load directory provided by no checkpoint found'



                  coord = tf.train.Coordinator()
                  threads = tf.train.start_queue_runners(sess=sess, coord=coord)



                  save_path_folder_matrix_train = os.path.join(FLAGS.log_dir, 'output_avg_features',  str(group),
                                                               str(block), str(conv),'train')


                  if not os.path.exists(save_path_folder_matrix_train) or FLAGS.recompute:
                      print('Getting training set...')
                      if not os.path.exists(save_path_folder_matrix_train):
                          os.makedirs(save_path_folder_matrix_train)

                      val_batch_size = FLAGS.val_batch_size
                      n_val_batch = int(n_train_samples / val_batch_size)
                      train_output_avg = np.zeros((n_train_samples,s[3]), dtype=np.float32)
                      train_labels = np.zeros((n_train_samples), dtype=np.int64)
                      for i in range(n_val_batch):
                          fetches = [output_avg, label_batch]
                          session_outputs = sess.run(fetches,{training_set.name: True})
                          train_output_avg[i * val_batch_size:(i + 1) * val_batch_size, :] = session_outputs[0]
                          train_labels[i * val_batch_size:(i + 1) * val_batch_size] = session_outputs[1]
                      np.savez( os.path.join(save_path_folder_matrix_train,'file.npz'),train_labels,train_output_avg)


                  save_path_folder_matrix_test = os.path.join(FLAGS.log_dir, 'output_avg_features',  str(group),
                                                              str(block), str(conv),'test')
                  if not os.path.exists(save_path_folder_matrix_test) or FLAGS.recompute:
                      print('Getting testing set...')
                      if not os.path.exists(save_path_folder_matrix_test):
                          os.makedirs(save_path_folder_matrix_test)

                      val_batch_size = FLAGS.val_batch_size
                      n_val_batch = int(n_val_samples / val_batch_size)
                      test_output_avg = np.zeros((n_val_samples,s[3]), dtype=np.float32)
                      test_labels = np.zeros((n_val_samples), dtype=np.int64)
                      for i in range(n_val_batch):
                          fetches = [output_avg, label_batch]
                          session_outputs = sess.run(fetches,{training_set.name: False})
                          test_output_avg[i * val_batch_size:(i + 1) * val_batch_size, :] = session_outputs[0]
                          test_labels[i * val_batch_size:(i + 1) * val_batch_size] = session_outputs[1]

                      np.savez(os.path.join(save_path_folder_matrix_test,'file.npz'), test_labels, test_output_avg)  # extraire training set avec un sess.run

                  save_file=os.path.join(FLAGS.log_dir, 'output_avg_features', str(group),
                                         str(block), str(conv), 'file.npz')
                  if not os.path.isfile(save_file) or FLAGS.recompute:

                      ### Loading data
                      print(os.path.join(save_path_folder_matrix_train,'file.npz'))
                      data=np.load(os.path.join(save_path_folder_matrix_train,'file.npz'))
                      train_labels=data['arr_0']
                      train_output_avg=data['arr_1']

                      #train_output_avg=train_output_avg[1:1000,:]
                      #train_labels = train_labels[1:1000]


                      data = np.load(os.path.join(save_path_folder_matrix_test, 'file.npz'))
                      test_labels = data['arr_0']
                      test_output_avg = data['arr_1']

                      ### Scaling data - NOT NECESSARY
                      scaler = preprocessing.StandardScaler().fit(train_output_avg)
                      train_output_avg=scaler.transform(train_output_avg)
                      test_output_avg = scaler.transform(test_output_avg)


                      ### Computing torsion
                      for c in range(10):
                          train_c=train_output_avg[train_labels == c, :]
                          D=pairwise_distances(train_c,n_jobs=-1)
                          torsion[layer][c]=np.sum(D.flatten())
                          #e = np.linalg.eig(np.cov(train_output_avg[train_labels == c, :].transpose()))
                          #PCA[layer, c, :] = np.squeeze(e[0])

                      ### Estimate linear contraction

                      for c in range(10):
                        e=np.linalg.eig(np.cov(train_output_avg[train_labels==c,:].transpose()))
                        PCA[layer,c,:]=np.squeeze(e[0])



                      ### 1-NN
                      neigh = KNeighborsClassifier(n_neighbors=1,n_jobs=-1)#,algorithm='brute')
                      neigh.fit(train_output_avg, train_labels)  # [support_,:], train_labels[support_])
                      predict_test = neigh.predict(test_output_avg)
                      test_accuracy = np.count_nonzero(predict_test == test_labels) / n_val_samples
                      acc_NN[layer]=test_accuracy
                      print('1NN: test acc : %s' % test_accuracy)

                      ### SVM
                      d = train_output_avg.shape[1]
                      radius = np.sqrt(d)
                      gam = 1 / (2 * radius * radius)
                      neigh = SVC(gamma=gam)
                      neigh.fit(train_output_avg, train_labels)  # [support_,:], train_labels[support_])
                      predict_test = neigh.predict(test_output_avg)
                      test_accuracy = np.count_nonzero(predict_test == test_labels) / n_val_samples
                      acc_SVM[layer] = test_accuracy
                      print('SVM: test acc : %s' % test_accuracy)

                      ### Histograms of distances of a 2-NN
                      nghbr = NearestNeighbors(n_neighbors=2)
                      nghbr.fit(train_output_avg)

                      dist, idx = nghbr.kneighbors(train_output_avg)
                      correct_idx=train_labels==train_labels[idx[:,1]]
                      incorrect_idx = train_labels != train_labels[idx[:,1]]
                      corr_dist=np.squeeze(dist[correct_idx, 1])
                      incorr_dist=np.squeeze(dist[incorrect_idx, 1])

                      hist.append(corr_dist)
                      hist.append(incorr_dist)


                      ### Liptschitz factor
                      L.append(np.max(np.linalg.norm(train_output_avg, axis=1)))

                      ### Estimation of the complexity of the boundary
                      SV = np.ones((50000), dtype=bool)
                      tmp = train_output_avg
                      tmp_lab = train_labels
                      for i in range(30):
                        neighs = KNeighborsClassifier(n_neighbors=i + 2, n_jobs=-1)
                        neighs.fit(train_output_avg,
                                   train_labels)  # [SV,:], train_labels[SV])  # [support_,:], train_labels[support_])
                        tmp = tmp[SV, :]
                        tmp_lab = tmp_lab[SV]
                        SV = neighs.predict(tmp) != tmp_lab
                        print('number of SV via %d-NN: %f' % (i + 2, np.sum(SV)))
                        N_SV[layer, i] = np.sum(SV)
                  layer = layer + 1
  np.save('acc_NN.npy', acc_NN)
  np.save('acc_SVM.npy', acc_SVM)
  np.save('incorrect_renorm.npy',hist)
  np.save('spec.npy',PCA)
  np.save('lipschitz.npy', L)
  np.save('torsion.npy', torsion)
  np.save('SVs.npy', N_SV)

  """







                      # Estimate volume


                      # Estimate complexity of support vector




                  coord.request_stop()
                  coord.join(threads)
                  sess.close()


                      # scaling data
                      scaler = preprocessing.StandardScaler().fit(train_output_avg)
                      train_output_avg=scaler.transform(train_output_avg)
                      test_output_avg = scaler.transform(test_output_avg)


                      d=train_output_avg.shape[1]
                      #print(d)
                      radius=np.sqrt(d)
                      gam=1/(2*radius*radius)





                      neighs = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
                      neighs.fit(train_output_avg, train_labels)
                      predict_train = neighs.predict(train_output_avg)
                      [dist,_]=neighs.kneighbors()
                      np.save('distance'+str(N)+'.npy',dist)
                      np.save('labels'+str(N)+'.npy',predict_train==train_labels)
                      N = N + 1

  np.save('mesSV.npz',N_SV)
  n = 0
  good_classif = []
  name = []
  support_vect = []
  # Now we build Gamma_n
  for group in range(3):
    for block in range(2):
      for conv in range(2):
        name.append('res_net/group_' + str(group + 1) + '/block_' + str(block + 1) + '/nonlinearity_' + str(
          conv + 1) + "/concat")
        save_file = os.path.join(FLAGS.log_dir, 'output_avg_features', str(group),
                                 str(block), str(conv), 'file.npz')
        data = np.load(save_file)
        test_labels = data['arr_0']
        train_labels = data['arr_1']
        predict_train = data['arr_2']
        predict_test = data['arr_3']
        SV_idx=data['arr_4']


        support_vect.append(SV_idx)
        good_classif.append(predict_train == train_labels)
  np.save('dagamma.npz', good_classif)
# Let's build auxiliary dataset
  GAMMA = []
  GAMMA_newly_classified=[]
  dataset_name = []
  dataset_newly_name = []
  for i in range(10):
    GAMMA.append(np.logical_and(good_classif[i], good_classif[i + 1]))
    GAMMA_newly_classified.append(np.logical_and(np.logical_not(good_classif[i]), good_classif[i + 1]))
    print('bien clkassif :%f'%np.sum(good_classif[i]))
    print('il y a gamma_n  inter gamma n+1 : %f' % np.sum(GAMMA[i]))
    print('il y a gamma_n ^c inter gamma n+1 : %f'%np.sum(GAMMA_newly_classified[i]))
    print('il y a suopport vect : %f'%np.size(np.unique(support_vect[i+1])))
    print('il y a en intersectin : %f'%np.size(np.unique(np.intersect1d(support_vect[i],support_vect[i+1]))))
    folder_data = os.path.join(FLAGS.log_dir, 'data_layer_' + str(i))
    if not os.path.exists(folder_data) or FLAGS.recompute:
      if not os.path.exists(folder_data):
        os.makedirs(folder_data)
      if not os.path.exists(folder_data+'lol'):
        os.makedirs(folder_data+'lol')
      create_dataset.create_trainset_with_mask(GAMMA[i], folder_data)
      create_dataset.create_trainset_with_mask(GAMMA_newly_classified[i], folder_data+'lol')
    dataset_name.append(os.path.join(folder_data, 'train.tf'))
    dataset_newly_name.append(os.path.join(folder_data+'lol', 'train.tf'))
  #GAMMA = np.array(GAMMA)
  #GAMMA_newly_classified = np.array(GAMMA_newly_classified)"""




if __name__ == '__main__':
    build_statistics()
