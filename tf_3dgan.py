# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting
import h5py

from keras.layers.convolutional import UpSampling3D, ZeroPadding3D
# load the data
with h5py.File('full_dataset_vectors.h5', 'r') as hf:
    x_train_raw = hf["X_train"][:]
    y_train_raw = hf["y_train"][:]
    x_test_raw = hf["X_test"][:]
    y_test_raw = hf["y_test"][:]


# Transform data from 1d to 3d rgb
def data_transform(data):
    result = np.zeros(25 * 25 * 25)
    data_t = []
    for i in range(data.shape[0]):
        result[:data[i].shape[0]] = data[i]
        data_t.append(result.reshape(25, 25, 25, 1))
    return np.asarray(data_t, dtype=np.float32)

n_classes = 1

x_train = data_transform(x_train_raw)
x_test = data_transform(x_test_raw)
x_train_data=x_train


def generator(z,reuse=None):
    with tf.variable_scope('gen',reuse=reuse):
        hidden1 = tf.layers.dense(inputs=z, units=64 * 7 * 7)

        reshape1=tf.reshape(hidden1, [-1, 8, 7, 7, 8])
        
        hidden2=tf.layers.conv3d(inputs=reshape1, filters=64, kernel_size=(6, 6, 8), padding='same', kernel_initializer=tf.keras.initializers.he_uniform(), activation=tf.nn.leaky_relu)
        
        batchnorm1=tf.layers.batch_normalization(hidden2)
        
        upsampling1=tf.keras.layers.UpSampling3D(size=(2, 2, 2))(batchnorm1)
        
        zeropadding1=tf.keras.layers.ZeroPadding3D((2, 2, 0))(upsampling1)
        
        hidden3=tf.layers.conv3d(inputs=zeropadding1, filters=16, kernel_size=(6, 5, 8), kernel_initializer=tf.keras.initializers.he_uniform(), activation=tf.nn.leaky_relu)
        
        batchnorm2=tf.layers.batch_normalization(hidden3)
        
        upsampling2=tf.keras.layers.UpSampling3D(size=(2, 2, 3))(batchnorm2)

        zeropadding2=tf.keras.layers.ZeroPadding3D((1, 0, 3))(upsampling2)

        hidden4=tf.layers.conv3d(inputs=zeropadding2, filters=16, kernel_size=(3, 3, 8), kernel_initializer=tf.keras.initializers.he_uniform(), activation=tf.nn.leaky_relu)
        
        hidden5=tf.layers.conv3d(inputs=hidden4, filters=1, kernel_size=(2, 2, 2), kernel_initializer=tf.keras.initializers.glorot_normal(), activation=tf.nn.relu, use_bias=False)

        return hidden5


def discriminator(X,reuse=None):
    print("X shape", X.shape)
    with tf.variable_scope('dis',reuse=reuse):
        hidden1=tf.layers.conv3d(inputs=X, filters=16, kernel_size=(5, 5, 5), padding='same', activation=tf.nn.leaky_relu)
        
        dropout1=tf.layers.dropout(hidden1, 0.2)
        
        zeropadding1=tf.keras.layers.ZeroPadding3D((2, 2, 2))(dropout1)
        
        hidden2=tf.layers.conv3d(inputs=zeropadding1, filters=16, kernel_size=(5, 5, 5), padding='valid', activation=tf.nn.leaky_relu)
        
        batchnorm1=tf.layers.batch_normalization(hidden2)
        
        dropout2=tf.layers.dropout(batchnorm1, 0.2)

        zeropadding2=tf.keras.layers.ZeroPadding3D((2, 2, 2))(dropout2)
        
        hidden3=tf.layers.conv3d(inputs=zeropadding2, filters=16, kernel_size=(5, 5, 5), padding='valid', activation=tf.nn.leaky_relu)
        
        batchnorm2=tf.layers.batch_normalization(hidden3)
        
        dropout3=tf.layers.dropout(batchnorm2, 0.2)
        
        zeropadding3=tf.keras.layers.ZeroPadding3D((1, 1, 1))(dropout3)
        
        hidden4=tf.layers.conv3d(inputs=zeropadding3, filters=16, kernel_size=(5, 5, 5), padding='valid', activation=tf.nn.leaky_relu)
        
        batchnorm3=tf.layers.batch_normalization(hidden4)
        
        dropout4=tf.layers.dropout(batchnorm3, 0.2)

        avgpooling=tf.layers.average_pooling3d(dropout4, (2, 2, 2), 1)

        flatten=tf.layers.flatten(avgpooling)

        fake=tf.layers.dense(flatten, units=1, activation=tf.nn.sigmoid)
        
        aux=tf.layers.dense(flatten, units=1, activation=None)
        
        ecal=tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=(2, 3, 4)))(X)

        return fake,aux
    

tf.reset_default_graph()

real_images=tf.placeholder(tf.float32,shape=[None, 25, 25, 25, 1])
z=tf.placeholder(tf.float32,shape=[None,100])

G=generator(z)
D_output_real,D_logits_real=discriminator(real_images)
D_output_fake,D_logits_fake=discriminator(G,reuse=True)

def loss_func(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))

D_real_loss=loss_func(D_logits_real,tf.ones_like(D_logits_real)*0.9) #Smoothing for generalization
D_fake_loss=loss_func(D_logits_fake,tf.zeros_like(D_logits_real))
D_loss=D_real_loss+D_fake_loss

G_loss= loss_func(D_logits_fake,tf.ones_like(D_logits_fake))

lr=0.001

#Do this when multiple networks interact with each other
tvars=tf.trainable_variables()  #returns all variables created(the two variable scopes) and makes trainable true
d_vars=[var for var in tvars if 'dis' in var.name]
g_vars=[var for var in tvars if 'gen' in var.name]

D_trainer=tf.train.AdamOptimizer(lr).minimize(D_loss,var_list=d_vars)
G_trainer=tf.train.AdamOptimizer(lr).minimize(G_loss,var_list=g_vars)

batch_size=100
epochs=5
init=tf.global_variables_initializer()

samples=[] #generator examples

with tf.Session() as sess:
    sess.run(init)
    num_batches = int(len(x_train_data)/batch_size) + 1
    for epoch in range(epochs):

        epoch_lossG = 0
        epoch_lossD = 0
        for i in range(num_batches):
            batch_images = x_train_data[i*batch_size: (i+1)*batch_size]
            if batch_images.shape[0]>0:
                batch_images=batch_images.reshape((batch_size, 25, 25, 25, 1))
                batch_images=batch_images*2-1
                batch_z=np.random.uniform(-1,1,size=(batch_size,100))
                _ = sess.run(D_trainer,feed_dict={real_images:batch_images,z:batch_z})
                _=sess.run(G_trainer,feed_dict={z:batch_z})

        print("on epoch{}".format(epoch))
        
        sample_z=np.random.uniform(-1,1,size=(1,100))
        gen_sample=sess.run(generator(z,reuse=True),feed_dict={z:sample_z})
        
        samples.append(gen_sample)


# # plt.imshow(samples[0][0].reshape(16,16,16,3))

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# samples=samples[0][0].reshape(16,16,16)
# import ipdb; ipdb.set_trace()

# ax.plot(*samples, label='parametric curve')
# ax.legend()

# plt.show()
# # plt.imshow(samples[20][0].reshape(16,16,16,3))