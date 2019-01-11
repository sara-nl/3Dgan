# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.framework import ops
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
import time
import glob
import math
import psutil
import socket
import horovod.tensorflow as hvd
from keras.layers.convolutional import UpSampling3D, ZeroPadding3D
import os
try:
    import numpy.random_intel as rng
except ImportError:
    from numpy import random as rng


def bit_flip(x, prob=0.05):
    x = np.array(x)
    selection = rng.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


config = tf.ConfigProto(log_device_placement=False)
config.intra_op_parallelism_threads = 11
config.inter_op_parallelism_threads = 1
os.environ['KMP_BLOCKTIME'] = str(1)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
os.environ['KMP_AFFINITY'] = 'balanced'
os.environ['OMP_NUM_THREADS'] = str(11)

hvd.init()
batch_size = 128
latent_size = 200
epochs = 5
g_batch_size = batch_size * hvd.size()


def DivideFiles(FileSearch="/data/LCD/*/*.h5", nEvents=200000, EventsperFile = 10000, Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    
    Files =sorted( glob.glob(FileSearch))
    Filesused = int(math.ceil(nEvents/EventsperFile))
    FileCount=0
   
    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")

        if ParticleName in Particles:
            try:
                Samples[ParticleName].append(F)
            except:
                Samples[ParticleName]=[(F)]

        if MaxFiles>0:
            if FileCount>MaxFiles:
                break
    out=[]
    for j in range(len(Fractions)):
        out.append([])

    SampleI=len(Samples.keys())*[int(0)]

    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName][:Filesused]
        NFiles=len(Sample)

        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI

    return out


# This functions loads data from a file and also does any pre processing
def GetData(datafile, xscale =1, yscale = 100, dimensions = 3):
    #get data for training
    try:
        if hvd.rank()==0:
            print('Loading Data from .....', datafile)  
    except NameError as e:
        print('Loading Data from .....', datafile)
        print('Running without horovod support')                                          
                                
    f=h5py.File(datafile,'r')
    
    X=np.array(f.get('ECAL'))
    
    Y=f.get('target')
    Y=(np.array(Y[:,1]))

    X[X < 1e-6] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    if dimensions == 2:
        X = np.sum(X, axis=(1))
        X = xscale * X
    # X = np.moveaxis(X, -1, 1)

    Y = np.expand_dims(Y, axis=-1)
    Y = Y.astype(np.float32)
    Y = Y/yscale
    # Y = np.moveaxis(Y, -1, 1)

    ecal = np.sum(X, axis=(2, 3, 4))
    
    return X, Y, ecal


# Transform data from 1d to 3d rgb
def data_transform(data):
    result = np.zeros(25 * 25 * 25)
    data_t = []
    # print(data.shape)
    for i in range(data.shape[0]):
        # print(data[i].shape[0])
        result[:data[i].shape[0]] = data[i]
        data_t.append(result.reshape(25, 25, 25, 1))
    return np.asarray(data_t, dtype=np.float32)


Trainfiles, Testfiles = DivideFiles("/scratch/04653/damianp/eos/project/d/dshep/LCD/V1/*scan/*.h5", nEvents=200000, EventsperFile=10000, datasetnames=["ECAL"], Particles=["Ele"], MaxFiles=1)

if hvd.rank()==0:
    print("Train files: {0} \nTest files: {1}".format(Trainfiles, Testfiles))

#Read test data into a single array
for index, dtest in enumerate(Testfiles):
   if index == 0:
       X_test, Y_test, ecal_test = GetData(dtest)
   else:
       X_temp, Y_temp, ecal_temp = GetData(dtest)
       X_test = np.concatenate((X_test, X_temp))
       Y_test = np.concatenate((Y_test, Y_temp))
       ecal_test = np.concatenate((ecal_test, ecal_temp))

for index, dtrain in enumerate(Trainfiles):
    if index == 0:
        X_train, Y_train, ecal_train = GetData(dtrain)
    else:
        X_temp, Y_temp, ecal_temp = GetData(dtrain)
        X_train = np.concatenate((X_train, X_temp))
        Y_train = np.concatenate((Y_train, Y_temp))
        ecal_train = np.concatenate((ecal_train, ecal_temp))


print("On hostname {0} - After init using {1} memory".format(socket.gethostname(), psutil.Process(os.getpid()).memory_info()[0]))

# X_train = data_transform(X_temp)
# Y_train = data_transform(Y_train)
# ecal_train = data_transform(ecal_train)

def generator(z,reuse=None):
    with tf.variable_scope('gen',reuse=reuse):
        hidden1 = tf.layers.dense(inputs=z, units=64 * 7 * 7)
        reshape1=tf.reshape(hidden1, [-1, 7, 7, 8, 8])
        hidden2=tf.layers.conv3d(inputs=reshape1, filters=64, kernel_size=(6, 6, 8), padding='same', kernel_initializer=tf.keras.initializers.he_uniform(), activation=tf.nn.leaky_relu)
        batchnorm1=tf.layers.batch_normalization(hidden2)
        
        upsampling1=tf.keras.layers.UpSampling3D(size=(2, 2, 2))(batchnorm1)
        zeropadding1=tf.keras.layers.ZeroPadding3D((2, 2, 0))(upsampling1)
        
        hidden3=tf.layers.conv3d(inputs=zeropadding1, filters=6, kernel_size=(6, 5, 8), kernel_initializer=tf.keras.initializers.he_uniform(), activation=tf.nn.leaky_relu)
        batchnorm2=tf.layers.batch_normalization(hidden3)
        
        upsampling2=tf.keras.layers.UpSampling3D(size=(2, 2, 3))(batchnorm2)
        zeropadding2=tf.keras.layers.ZeroPadding3D((1, 0, 3))(upsampling2)

        hidden4=tf.layers.conv3d(inputs=zeropadding2, filters=6, kernel_size=(3, 3, 8), kernel_initializer=tf.keras.initializers.he_uniform(), activation=tf.nn.leaky_relu)
        hidden5=tf.layers.conv3d(inputs=hidden4, filters=1, kernel_size=(2, 2, 2), kernel_initializer=tf.keras.initializers.glorot_normal(), activation=tf.nn.relu, use_bias=False)

        return hidden5


def discriminator(X,reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        hidden1=tf.layers.conv3d(inputs=X, filters=32, kernel_size=(5, 5, 5), padding='same', activation=tf.nn.leaky_relu)
        dropout1=tf.layers.dropout(hidden1, 0.2)
        
        zeropadding1=tf.keras.layers.ZeroPadding3D((2, 2, 2))(dropout1)
        
        hidden2=tf.layers.conv3d(inputs=zeropadding1, filters=8, kernel_size=(5, 5, 5), padding='valid', activation=tf.nn.leaky_relu)
        batchnorm1=tf.layers.batch_normalization(hidden2)
        dropout2=tf.layers.dropout(batchnorm1, 0.2)

        zeropadding2=tf.keras.layers.ZeroPadding3D((2, 2, 2))(dropout2)
        
        hidden3=tf.layers.conv3d(inputs=zeropadding2, filters=8, kernel_size=(5, 5, 5), padding='valid', activation=tf.nn.leaky_relu)
        batchnorm2=tf.layers.batch_normalization(hidden3)
        dropout3=tf.layers.dropout(batchnorm2, 0.2)
        
        zeropadding3=tf.keras.layers.ZeroPadding3D((1, 1, 1))(dropout3)
        
        hidden4=tf.layers.conv3d(inputs=zeropadding3, filters=8, kernel_size=(5, 5, 5), padding='valid', activation=tf.nn.leaky_relu)
        batchnorm3=tf.layers.batch_normalization(hidden4)
        dropout4=tf.layers.dropout(batchnorm3, 0.2)
        avgpooling=tf.layers.average_pooling3d(dropout4, (2, 2, 2), 1)
        flatten=tf.layers.flatten(avgpooling)
        
        fake=tf.layers.dense(flatten, units=1, activation=tf.nn.sigmoid)
        aux=tf.layers.dense(flatten, units=1, activation=None)
        ecal=tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=(1, 2, 3)))(X)

        return fake, aux, ecal
    

tf.reset_default_graph()

real_images = tf.placeholder(tf.float32,shape=[None, 25, 25, 25, 1])
z = tf.placeholder(tf.float32,shape=[None, latent_size])

G=generator(z)
D_real_output, D_real_aux, D_real_ecal = discriminator(real_images)
D_fake_output, D_fake_aux, D_fake_ecal = discriminator(G,reuse=True)

def binary_crossentropy(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_in, logits=logits_in))

def mean_absolute_percentage_error(outputs,y):
    return tf.reduce_mean(tf.abs(tf.divide(tf.subtract(outputs,y),y)))

D_real_output_loss = binary_crossentropy(D_real_output, tf.ones_like(D_real_output)) #*0.9)
D_real_aux_loss = mean_absolute_percentage_error(D_real_aux, tf.ones_like(D_real_aux)) #*0.9)
D_real_ecal_loss = mean_absolute_percentage_error(D_real_ecal,tf.ones_like(D_real_ecal)) #*0.9)
D_real_loss = tf.reduce_sum([6*D_real_output_loss, 0.2*D_real_aux_loss, 0.1*D_real_ecal_loss])

D_fake_output_loss = binary_crossentropy(D_real_output, tf.zeros_like(D_real_output))
D_fake_aux_loss = mean_absolute_percentage_error(D_fake_aux, tf.zeros_like(D_fake_aux))
D_fake_ecal_loss = mean_absolute_percentage_error(D_fake_ecal, tf.zeros_like(D_fake_ecal))
D_fake_loss = tf.reduce_sum([6*D_fake_output_loss, 0.2*D_fake_aux_loss, 0.1*D_fake_ecal_loss])

D_loss = (D_real_output_loss + D_fake_loss) / 2

G_loss = binary_crossentropy(D_fake_output, tf.ones_like(D_fake_output))

lr=0.001

#Do this when multiple networks interact with each other
tvars=tf.trainable_variables()  #returns all variables created(the two variable scopes) and makes trainable true
d_vars=[var for var in tvars if 'dis' in var.name]
g_vars=[var for var in tvars if 'gen' in var.name]

D_trainer=hvd.DistributedOptimizer(tf.train.AdamOptimizer(lr * hvd.size())).minimize(D_loss,var_list=d_vars)
G_trainer=hvd.DistributedOptimizer(tf.train.AdamOptimizer(lr * hvd.size())).minimize(G_loss,var_list=g_vars)

samples=[] #generator examples

# all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
# print(all_variables_list)
# tf.variables_initializer(var_list=all_variables_list)
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(init)
    hvd.broadcast_global_variables(0)
    num_batches = int(len(X_train)/g_batch_size) + 1
    
    for epoch in range(epochs):
        startt=time.time()
        epoch_lossG = 0
        epoch_lossD = 0

        for i in range(num_batches):
            print("doing batch {}".format(i))
            noise = rng.normal(0, 1, (batch_size, latent_size))
            image_batch = X_train[i*batch_size: (i+1)*batch_size]
            energy_batch = Y_train[i*batch_size: (i+1)*batch_size].flatten()
            ecal_batch = ecal_train[i*batch_size: (i+1)*batch_size]

            if image_batch.shape[0]>0:
                sampled_energies = rng.uniform(1, 5, size=(batch_size, 1))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = np.multiply(2, sampled_energies)

                _ = sess.run(G_trainer,feed_dict={z:generator_ip})

                generated=sess.run(generator(z,reuse=True),feed_dict={z:generator_ip})

                _ = sess.run(D_trainer,feed_dict={real_images:image_batch,z:generator_ip})
                _ = sess.run(D_trainer,feed_dict={real_images:generated,z:generator_ip})

                trick = np.ones(batch_size)

                for _ in range(2):
                    noise = rng.normal(0, 1, (batch_size, latent_size))
                    sampled_energies = rng.uniform(1, 5, ( batch_size,1 ))
                    generator_ip = np.multiply(sampled_energies, noise)
                    ecal_ip = np.multiply(2, sampled_energies)
                    _ = sess.run(D_trainer,feed_dict={real_images:generated,z:generator_ip})
                

        print("Epoch{} took {}s".format(epoch, time.time()-startt))

        sample_z=rng.uniform(1, 5,size=(batch_size,latent_size))
        starti=time.time()
        gen_sample=sess.run(generator(z,reuse=True),feed_dict={z:sample_z})
        print("Generation for 1 sample took {}s".format((time.time()-starti)/batch_size))
        samples.append(gen_sample)