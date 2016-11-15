from __future__ import division

import sys, os, time, math
import ipdb
import pickle
import tensorflow as tf
import joblib
import numpy as np

from scipy.io import loadmat

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '', 'Example: ./data_cifar10/cifar-10-batches-py/')
tf.app.flags.DEFINE_string('data_name', '', 'Example: cifar10')

# data_root = '/users/data/oyallon/resnettf/tf_resnet_cifar/mywork/src/cifar10_data/cifar-10-batches-py/'
def create_dataset(data_root,data_name):
    def save_to_records(save_path, images, labels):
        writer = tf.python_io.TFRecordWriter(save_path)
        for i in range(images.shape[0]):
            image_raw = images[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(32),
                'width': _int64_feature(32),
                'depth': _int64_feature(3),
                'label': _int64_feature(int(labels[i])),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())

    # train set

    train_images = np.zeros((50000,3072), dtype=np.uint8)
    trian_labels = np.zeros((50000,), dtype=np.int32)
    if(data_name=='cifar10'):
        for i in range(5):
            with open(os.path.join(data_root, 'data_batch_%d' % (i+1)),"rb") as f:
                data_batch=pickle.load(f,encoding='bytes')
                train_images[10000*i:10000*(i+1)] = data_batch[b'data']
                trian_labels[10000*i:10000*(i+1)] = np.asarray(data_batch[b'labels'], dtype=np.int32)
        train_images = np.reshape(train_images, [50000,3,32,32])
        train_images = np.transpose(train_images, axes=[0,2,3,1]) # NCHW -> NHWC
        save_to_records('data_'+data_name+'/train.tf', train_images, trian_labels)
    elif(data_name=='cifar100'):
        with open(os.path.join(data_root, 'train'), "rb") as f:
            data_batch = pickle.load(f, encoding='bytes')
            train_images[:,:] = data_batch[b'data']
            trian_labels[:] = np.asarray(data_batch[b'fine_labels'], dtype=np.int32)
        train_images = np.reshape(train_images, [50000, 3, 32, 32])
        train_images = np.transpose(train_images, axes=[0, 2, 3, 1])  # NCHW -> NHWC
        save_to_records('data_' + data_name + '/train.tf', train_images, trian_labels)
    # mean and std
    image_mean = np.mean(train_images.astype(np.float32), axis=(0,1,2))
    image_std = np.std(train_images.astype(np.float32), axis=(0,1,2))
    joblib.dump({'mean': image_mean, 'std': image_std}, 'data_'+data_name+'/meanstd.pkl', compress=5)

    # test set
    if(data_name=='cifar10'):
        with open(os.path.join(data_root, 'test_batch'),"rb") as f:
            data_batch=pickle.load(f,encoding='bytes')
        #  data_batch = joblib.load(os.path.join(data_root, 'test_batch'))
            test_images = data_batch[b'data']
            test_images = np.reshape(test_images, [10000,3,32,32])
            test_images = np.transpose(test_images, axes=[0,2,3,1])
            test_labels = np.asarray(data_batch[b'labels'], dtype=np.int32)
            save_to_records('data_'+data_name+'/test.tf', test_images, test_labels)
    elif(data_name=='cifar100'):
        with open(os.path.join(data_root, 'test'), "rb") as f:
            data_batch = pickle.load(f, encoding='bytes')
            #  data_batch = joblib.load(os.path.join(data_root, 'test_batch'))
            test_images = data_batch[b'data']
            test_images = np.reshape(test_images, [10000, 3, 32, 32])
            test_images = np.transpose(test_images, axes=[0, 2, 3, 1])
            test_labels = np.asarray(data_batch[b'fine_labels'], dtype=np.int32)
            save_to_records('data_' + data_name + '/test.tf', test_images, test_labels)


# data_root = '/users/data/oyallon/resnettf/tf_resnet_cifar/mywork/src/cifar10_data/cifar-10-batches-py/'
def create_trainset_with_mask(mask,wheretosave,date_root):
    def save_to_records(save_path, images, labels):
        writer = tf.python_io.TFRecordWriter(save_path)
        for i in range(images.shape[0]):
            image_raw = images[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(32),
                'width': _int64_feature(32),
                'depth': _int64_feature(3),
                'label': _int64_feature(int(labels[i])),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())

    # train set

    train_images = np.zeros((50000, 3072), dtype=np.uint8)
    trian_labels = np.zeros((50000,), dtype=np.int32)
    for i in range(5):
        with open(os.path.join(data_root, 'data_batch_%d' % (i + 1)), "rb") as f:
            data_batch = pickle.load(f, encoding='bytes')

            #    batch_file=os.path.join(data_root, 'data_batch_%d' % (i+1))
            #    data_batch = unpickle(batch_file)#joblib.load(os.path.join(data_root, 'data_batch_%d' % (i+1)))

            train_images[10000 * i:10000 * (i + 1)] = data_batch[b'data']
            trian_labels[10000 * i:10000 * (i + 1)] = np.asarray(data_batch[b'labels'], dtype=np.int32)
    train_images = np.reshape(train_images, [50000, 3, 32, 32])
    train_images = np.transpose(train_images, axes=[0, 2, 3, 1])  # NCHW -> NHWC
    train_images = train_images[mask,:,:,:]
    trian_labels = trian_labels[mask]
    print(train_images.shape)
    save_to_records(os.path.join(wheretosave,'train.tf'), train_images, trian_labels)


if __name__ == '__main__':
    create_dataset(FLAGS.data_dir,FLAGS.data_name)
