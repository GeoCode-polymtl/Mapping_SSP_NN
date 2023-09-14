""" Group of functions defining multiple encoders and decoders, and methods for reading datasets with and without noise
 """
import os
import tensorflow as tf
import numpy as np
from importlib import import_module
from scipy.ndimage import gaussian_filter
from scipy import signal
import copy
from obspy.io.segy.segy import _read_segy
from obspy.signal.invsim import cosine_taper
import h5py as h5
import matplotlib
from hyperbolic_radon import hyperbolic_radon
from dispersion import dispersion
from tqdm import tqdm
# from GeoFlow.SeismicUtilities import random_noise
import tensorflow_addons as tfa
from tf_siren import ScaledSinusodialRepresentationDense, Sine, SIRENInitializer
# matplotlib.use('Agg')


# %% NN architectures
def encoder_disp():
    """Encoder for the dispersion plot"""
    l2 = 1e-7
    # encoder_input = tf.keras.layers.Input(shape=data_input.shape[1:])
    encoder_input = tf.keras.layers.Input(shape=(200,200,1),name='dispersion')
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input) # filters=64, kernel_size=(17,3)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                              strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x) # kernel_size=(17, 3)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x) # kernel_size=(17, 3)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x) # kernel_size=(9, 3)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x) # kernel_size=(5, 3)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    encoder_output = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x)
    return tf.keras.models.Model(encoder_input,encoder_output,name='encoder_disp')

def encoder_disp_siren():
    """Encoder for the dispersion plot"""
    l2 = 1e-5
    # encoder_input = tf.keras.layers.Input(shape=data_input.shape[1:])
    encoder_input = tf.keras.layers.Input(shape=(200,200,1),name='dispersion')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(17,3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer,bias_initializer=SIRENInitializer,
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(17, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(17, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(9, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    encoder_output = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x)
    return tf.keras.models.Model(encoder_input,encoder_output,name='encoder_disp_siren')

def encoder_disp2():
    """Encoder for the dispersion plot with recursive implementation"""
    l2 = 1e-7
    # encoder_input = tf.keras.layers.Input(shape=data_input.shape[1:])
    encoder_input = tf.keras.layers.Input(shape=(200,200,1),name='dispersion')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(17,3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1,1), kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    conv2d_8 = tf.keras.layers.Conv2D(filters=8, kernel_size=(17,3), activation=tf.nn.leaky_relu, padding='same',
                                      strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(l2))
    for l in range(2): x = conv2d_8(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(9, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    conv2d_16 = tf.keras.layers.Conv2D(filters=16, kernel_size=(9, 3), activation=tf.nn.leaky_relu, padding='same',
                                       strides=(2, 2), kernel_regularizer=tf.keras.regularizers.l2(l2))
    for l in range(2): x = conv2d_16(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    conv2d_32 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 3), activation=tf.nn.leaky_relu, padding='same',
                                       strides=(2, 2), kernel_regularizer=tf.keras.regularizers.l2(l2))
    for l in range(2): x = conv2d_32(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    conv2d_64 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                                       strides=(2, 2), kernel_regularizer=tf.keras.regularizers.l2(l2))
    for l in range(2): x = conv2d_64(x)
    encoder_output = x
    return tf.keras.models.Model(encoder_input,encoder_output,name='encoder_disp_recursive')

def encoder_sg():
    """ Encoder for the shot gather """
    l2 = 1e-7
    # encoder_input = tf.keras.layers.Input(shape=data_input.shape[1:])
    encoder_input = tf.keras.layers.Input(shape=(1000,120,1),name='shotgather')
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(17,3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input) # filters=64
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(5, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(5, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(9, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    encoder_output = tf.keras.layers.MaxPool2D(pool_size=(10, 7))(x)
    return tf.keras.models.Model(encoder_input,encoder_output,name='encoder_sg')

def encoder_sg_siren():
    """ Encoder for the shot gather """
    l2 = 1e-5
    encoder_input = tf.keras.layers.Input(shape=(1000,120,1),name='shotgather')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(17,3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(17, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(5, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(17, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(5, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(9, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    encoder_output = tf.keras.layers.MaxPool2D(pool_size=(10, 7))(x)
    return tf.keras.models.Model(encoder_input,encoder_output,name='encoder_sg_siren')

def encoder_sg2():
    """ Encoder for the shot gather with recursive implementation"""
    l2 = 1e-7
    # encoder_input = tf.keras.layers.Input(shape=data_input.shape[1:])
    encoder_input = tf.keras.layers.Input(shape=(1000,120,1),name='shotgather')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    conv2d_8 = tf.keras.layers.Conv2D(filters=8, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                                      strides=(5, 2), kernel_regularizer=tf.keras.regularizers.l2(l2))
    for l in range(2): x = conv2d_8(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(9, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    conv2d_16 = tf.keras.layers.Conv2D(filters=16, kernel_size=(9, 3), activation=tf.nn.leaky_relu, padding='same',
                                       strides=(2, 2), kernel_regularizer=tf.keras.regularizers.l2(l2))
    for l in range(2): x = conv2d_16(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    conv2d_32 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 3), activation=tf.nn.leaky_relu, padding='same',
                                       strides=(2, 2), kernel_regularizer=tf.keras.regularizers.l2(l2))
    for l in range(2): x = conv2d_32(x)
    encoder_output = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=tf.nn.leaky_relu, padding='same',
                                            strides=(3,2), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    return tf.keras.models.Model(encoder_input,encoder_output,name='encoder_sg_recursive')

def encoder_sg_rnn():
    """ RNN encoder for the shot gather """
    l2 = 1e-7
    # encoder_input = tf.keras.layers.Input(shape=data_input.shape[1:])
    encoder_input = tf.keras.layers.Input(shape=(1000,120,1),name='shotgather')
    'Reducing size in offset axis'
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(1,3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1,1), kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1,1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1,2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1,1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1,2))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1,1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1,2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1,3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1,1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1,7))(x)
    'RNN'
    rnn_input = tf.squeeze(x,axis=2)
    x = tf.keras.layers.LSTM(256,return_sequences=True,activation=tf.nn.leaky_relu)(rnn_input)
    rnn_output = tf.expand_dims(x,axis=2)
    'Reducing size in time axis'
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(17,1), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1,1), kernel_regularizer=tf.keras.regularizers.l2(l2))(rnn_output)
    x = tf.keras.layers.MaxPool2D(pool_size=(5,1))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(17,1), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1,1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(5,1))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(9,1), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1,1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,1))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(5,1), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1,1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,1))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,1), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1,1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    encoder_output = tf.keras.layers.MaxPool2D(pool_size=(10,1))(x)
    return tf.keras.models.Model(encoder_input, encoder_output, name='encoder_sg_RNN')

def encoder_fftradon():
    """Encoder for the radon in the frequency domain"""
    l2 = 1e-7
    # encoder_input = tf.keras.layers.Input(shape=data_input.shape[1:])
    encoder_input = tf.keras.layers.Input(shape=(200,200,1),name='fft_radon')
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input) # filters=64, kernel_size=(17,3)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                              strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x) # kernel_size=(17, 3)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x) # kernel_size=(17, 3)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x) # kernel_size=(9, 3)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x) # kernel_size=(5, 3)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                           strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    encoder_output = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x)
    return tf.keras.models.Model(encoder_input,encoder_output,name='encoder_fftradon')

def encoder_fftradon_siren():
    """Encoder for the radon in the frequency domain"""
    l2 = 1e-5
    encoder_input = tf.keras.layers.Input(shape=(200,200,1),name='fft_radon')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(17,3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(17, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(17, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(9, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    encoder_output = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x)
    return tf.keras.models.Model(encoder_input,encoder_output,name='encoder_fftradon_siren')

def encoder_radon():
    """Encoder for radon"""
    l2=1e-7
    # encoder_input = tf.keras.layers.Input(shape=data_input.shape[1:])
    encoder_input = tf.keras.layers.Input(shape=(1000,200,1),name='radon')
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input) # filters=64
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(4, 1))(x)
    # x = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(x)
    # x = tf.keras.layers.Conv2D(filters=8, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
    #                            strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    # x = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(9, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    encoder_output = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x)
    return tf.keras.models.Model(encoder_input, encoder_output, name='encoder_radon')

def encoder_radon_siren():
    """Encoder for radon"""
    l2=1e-5
    encoder_input = tf.keras.layers.Input(shape=(1000,200,1),name='radon')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(17, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(17, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(4, 1))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(17, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(17, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(9, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=Sine(), padding='same',
                               kernel_initializer=SIRENInitializer, bias_initializer=SIRENInitializer,
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    encoder_output = tf.keras.layers.MaxPool2D(pool_size=(3, 3))(x)
    return tf.keras.models.Model(encoder_input, encoder_output, name='encoder_radon_siren')


ly_conv2d_offset = lambda fil, l2: tf.keras.layers.Conv2D(filters=fil, kernel_size=(1, 3), activation=tf.nn.leaky_relu,
                                                          padding='same', strides=(1,2),
                                                          kernel_regularizer=tf.keras.regularizers.l2(l2))
ly_conv2d_time = lambda fil,l2,k: tf.keras.layers.Conv2D(filters=fil, kernel_size=(k + 1, 1),
                                                         activation=tf.nn.leaky_relu, padding='same',
                                                         strides=(2,1), kernel_regularizer=tf.keras.regularizers.l2(l2))

def encoder_sg_rnn2(data_input):
    """ Recursive RNN encoder for shot gather """
    l2 = 1e-7
    filters = 8
    encoder_input = tf.keras.layers.Input(shape=data_input.shape[1:])
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input)
    'Reducing size in offset axis'
    while x.get_shape()[2] > 1:
        ly_offset = ly_conv2d_offset(fil=filters, l2=l2)
        x = ly_offset(x)
        if filters < 64: filters *= 2
    # kernel = 16
    # filters = 8
    'RNN'
    rnn_input = tf.squeeze(x, axis=2)
    x = tf.keras.layers.LSTM(64, return_sequences=False, activation=tf.nn.leaky_relu)(rnn_input)
    x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=1)
    # 'Reducing size in time axis'
    # while x.get_shape()[1] >1:
    #     ly_time = ly_conv2d_time(fil=filters,l2=l2,k=kernel)
    #     x = ly_time(x)
    #     if filters < 256: filters *= 2
    #     if kernel > 2: kernel //= 2
    return tf.keras.models.Model(encoder_input, x, name='encoder_sg_RNN')

#def decoder_700m(encoder_output):
def decoder_700m_2lab(encoder_output):
    """ Decoder with 700m depth output for vp and vs """
    # decoder_input = tf.keras.layers.Input(shape=(1,1,256))
    decoder_input = tf.keras.layers.Input(shape=encoder_output.shape[1:])
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 2))(decoder_input)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(9, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(5, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(7, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(560, activation=tf.nn.leaky_relu)(x)
    decoder_output = tf.keras.layers.Reshape(target_shape=(280, 2))(x)
    return tf.keras.models.Model(decoder_input,decoder_output,name='decoder_gen_700m_2lab')

def decoder_700m_3lab(encoder_output):
    """ Decoder with 700m depth output for vp and vs """
    # decoder_input = tf.keras.layers.Input(shape=(1,1,256))
    decoder_input = tf.keras.layers.Input(shape=encoder_output.shape[1:])
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 2))(decoder_input)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(9, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(5, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(7, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(840, activation=tf.nn.leaky_relu)(x)
    decoder_output = tf.keras.layers.Reshape(target_shape=(280, 3))(x)
    return tf.keras.models.Model(decoder_input,decoder_output,name='decoder_gen_700m_3lab')

def decoder_700m(encoder_output,output_depth='700m',ol=['vp','vs','1/q'],name='decoder_gen_700m'):
    """ Decoder with generalized output and generalized number of out labels"""
    depth = int(output_depth[:-1])
    decoder_input = tf.keras.layers.Input(shape=encoder_output.shape[1:])
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 2))(decoder_input)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(9, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(7, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(int(depth/2.5*len(ol)), activation=tf.nn.leaky_relu)(x)
    decoder_output = tf.keras.layers.Reshape(target_shape=(int(depth/2.5), len(ol)))(x)
    return tf.keras.models.Model(decoder_input, decoder_output, name=name)

def decoder_700m_cat(encoder_output,output_depth='700m',ol=['vp','vs','1/q']):
    """ Decoder with generalized output and generalized number of out labels with categorical output"""
    depth = int(output_depth[:-1])
    decoder_input = tf.keras.layers.Input(shape=encoder_output.shape[1:])
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 2))(decoder_input)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(9, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(7, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(int(depth/2.5*len(ol)), activation=tf.nn.leaky_relu)(x)
    decoder_output = tf.keras.layers.Reshape(target_shape=(int(depth/2.5), len(ol),1))(x)
    decoder_output = tf.keras.layers.Conv2D(filters=100, kernel_size=(1,1), activation=tf.nn.softmax,
                                            padding='same',strides=(1, 1))(decoder_output)
    # decoder_output = tf.keras.layers.Softmax(axis=-1)(decoder_output)
    return tf.keras.models.Model(decoder_input, decoder_output, name='decoder_gen_700m_categorical')
def decoder_700m_vp(encoder_output):
    """ Decoder with 700m depth output for anyone, vp or vs (only one)"""
    decoder_input = tf.keras.layers.Input(shape=encoder_output.shape[1:])
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 1))(decoder_input)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(9, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(5, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(7, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(280, activation=tf.nn.leaky_relu)(x)
    decoder_output = tf.keras.layers.Reshape(target_shape=(280, 1))(x)
    return tf.keras.models.Model(decoder_input,decoder_output,name='decoder_gen_700m_vp')

@tf.function
def layered2grid(layered, dh, n, smooth=2):
    """
    :param layered: tensore with size [nbatch, n, nprop] with nbatch the
                    batch size, n is the number of layers, and nprop is the
                    number of properties to be interpolated. layered[:,:,0]
                    contains the layers thicknesses
    :param dh:      The grid spacing
    :param n:       Number of grid points
    :return:        Tensor holding the layered model in regular grid

    Parameters
    ----------
    smooth :
    """
    batch_size = tf.keras.backend.shape(layered)[0]
    posint = tf.tile(tf.reshape(tf.range(0, n, dtype=tf.float32), [1, -1, 1]) * dh,
                     [batch_size, 1, 1])
    d1 = tf.nn.softmax(layered[:,:,0:1])*700
    pos1 = tf.cumsum(d1, axis=1)[:, :-1, :] - smooth * dh
    pos2 = tf.cumsum(d1, axis=1) + smooth * dh

    # pos1 = tf.cumsum(layered[:, :, 0:1], axis=1)[:, :-1, :] - smooth * dh
    # pos2 = tf.cumsum(layered[:, :, 0:1], axis=1) + smooth * dh
    pos = tf.concat([tf.zeros_like(pos1[:, 0:1, :]),  pos1, pos2], axis=1)
    props1 = tf.concat([layered[:, :1, 1:], layered[:, :-1, 1:]], axis=1)
    props = tf.concat([props1, layered[:, 1:, 1:], layered[:, -1:, 1:]], axis=1)

    return tfa.image.interpolate_spline(pos, props, posint, order=1)

# @tf.function
# def layered2grid2(layered, dh, n, smooth=2):
#     # layer = tf.constant([[80, 1500, 0, 1000], [100, 3500, 2000, 20], [100, 2500, 1000, 80]])
#     # layered = tf.tile(tf.reshape(layer, [-1, *tf.shape(layer)]), [5, 1, 1])
#
#     dec_out_bl = tf.concat([tf.concat([tf.ones([1, layered[0, i, 0], int(tf.shape(layered)[-1]) - 1])
#                           * tf.cast(layered[j, i, 1:], dtype=tf.float32)
#                           for i in tf.range(int(tf.shape(layered)[-1]) - 1)], axis=1)[:, :n, :]
#                for j in tf.range(tf.shape(layered)[0])], axis=0)
#     return dec_out_bl
#
class Layer_layered2grid(tf.keras.layers.Layer):
    def __init__(self, dh, n, smooth=2):
        super().__init__()
        self.dh = dh
        self.n = n
        self.smooth = smooth

    def get_config(self):
        config = super().get_config()
        config.update({
            "dh": self.dh,
            "n": self.n,
            "smooth":self.smooth,
        })
        return config

    def call(self, inputs=None):
        return layered2grid(inputs, dh=self.dh, n=self.n, smooth=self.smooth)

def decoder_700m_blocky(encoder_output,output_depth='700m',ol=['vp','vs','1/q']):
    """ Decoder with 700m depth output for vp and vs """
    depth = int(output_depth[:-1])

    decoder_input = tf.keras.layers.Input(shape=encoder_output.shape[1:])
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 2))(decoder_input)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(9, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(5, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                    strides=(7, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(840, activation=tf.nn.leaky_relu)(x)
    t_s = (int(depth / 50), len(ol) + 1)
    x = tf.keras.layers.Dense(int(t_s[0]*t_s[1]), activation=tf.nn.leaky_relu)(x)
    decoder_output = tf.keras.layers.Reshape(target_shape=t_s)(x)
    dh = 2.5
    n = int(depth/dh)
    batch_size = tf.keras.backend.shape(decoder_output)[0]
    smooth = 2
    dec_out_blocky = Layer_layered2grid(dh,n,smooth)(decoder_output)
    # return tf.keras.models.Model(decoder_input,dec_out_blocky,name='decoder_gen_700m_blocky')

    # dec_out_blocky = tf.keras.layers.Conv1DTranspose(filters=3, kernel_size=1, activation=tf.nn.leaky_relu,
    #                                                  padding='same', strides=20)(decoder_output)
    #
    # x = tf.keras.layers.Conv1D(filters=3, kernel_size=1, activation=tf.nn.leaky_relu,padding='same')(decoder_output)
    # x = tf.keras.layers.Flatten()(x)
    # filt = int(depth/dh*len(ol))
    # x = tf.keras.layers.Dense(filt, activation=tf.nn.leaky_relu)(x)
    # dec_out_blocky = tf.keras.layers.Reshape(target_shape=(int(depth/dh),len(ol)))(x)

    return tf.keras.models.Model(decoder_input,dec_out_blocky,name='decoder_gen_700m_blocky')

def decoder_700m_blocky2(encoder_output,output_depth='700m',ol=['vp','vs','1/q']):
    """ Decoder with 700m depth output for vp and vs """
    depth = int(output_depth[:-1])

    decoder_input = tf.keras.layers.Input(shape=encoder_output.shape[1:])
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 1), activation=Sine(), padding='same',
                                        kernel_initializer=SIRENInitializer,strides=(2, 2))(decoder_input)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 1), activation=Sine(), padding='same',
                                        kernel_initializer=SIRENInitializer,strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(9, 1), activation=Sine(), padding='same',
                                        kernel_initializer=SIRENInitializer,strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(17, 1), activation=Sine(), padding='same',
                                        kernel_initializer=SIRENInitializer,strides=(5, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(17, 1), activation=Sine(), padding='same',
                                        kernel_initializer=SIRENInitializer,strides=(7, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = ScaledSinusodialRepresentationDense(int(depth/2.5*len(ol)),activation='sine')(x)
    decoder_output = tf.keras.layers.Reshape(target_shape=(int(depth/2.5),len(ol)))(x)
    return tf.keras.models.Model(decoder_input,decoder_output,name='decoder_gen_700m_blocky2')
def decoder_1000m(encoder_output):
    """ Decoder with 1000m depth output for vp and vs """
    # decoder_input = tf.keras.layers.Input(shape=(1,1,256))
    decoder_input = tf.keras.layers.Input(shape=encoder_output.shape[1:])
    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 2))(decoder_input)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(5, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(9, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(800, activation=tf.nn.leaky_relu)(x)
    decoder_output = tf.keras.layers.Reshape(target_shape=(400, 2))(x)
    return tf.keras.models.Model(decoder_input,decoder_output,name='decoder_gen_1000m')

def decoder_1000m_vp(encoder_output):
    """ Decoder with 1000m depth output for anyone, vp or vs (onlyone)"""
    decoder_input = tf.keras.layers.Input(shape=encoder_output.shape[1:])
    x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 1))(decoder_input)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(5, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(9, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(2, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 1))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(17, 1), activation=tf.nn.leaky_relu, padding='same',
                                        strides=(5, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(400, activation=tf.nn.leaky_relu)(x)
    decoder_output = tf.keras.layers.Reshape(target_shape=(400, 1))(x)
    return tf.keras.models.Model(decoder_input,decoder_output,name='decoder_gen_1000m_vp')

def res_conv(x,pool,filt,kernel):
    """Residual convolutional NN layer"""
    x_skip = x
    l2 = 1e-7
    x = tf.keras.layers.Conv2D(filters=filt, kernel_size=(1,1), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Conv2D(filters=filt, kernel_size=kernel, activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=pool)(x)
    x = tf.keras.layers.Conv2D(filters=filt, kernel_size=(1,1), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x_skip = tf.keras.layers.Conv2D(filters=filt, kernel_size=(1,1), activation=tf.nn.leaky_relu, padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2))(x_skip)
    x_skip = tf.keras.layers.MaxPool2D(pool_size=pool)(x_skip)
    x = tf.keras.layers.Add()([x,x_skip])
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def encoder_fftradon2():
    """Encoder for fftradon using res_conv"""
    l2 = 1e-7
    encoder_input = tf.keras.layers.Input(shape=(200,200,1),name='fft_radon')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(17, 3), activation=tf.nn.leaky_relu, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2))(encoder_input)
    x = res_conv(x, pool=(2, 2), filt=8,     kernel=(17, 3))
    x = res_conv(x, pool=(2, 2), filt=16,   kernel=(17, 3))
    x = res_conv(x, pool=(2, 2), filt=32,   kernel=(9, 3))
    x = res_conv(x, pool=(2, 2), filt=64,   kernel=(5, 3))
    x = res_conv(x, pool=(2, 2), filt=128, kernel=(3, 3))
    x = res_conv(x, pool=(2, 2), filt=256, kernel=(3, 3))
    encoder_output = res_conv(x, pool=(3, 3), filt=256, kernel=(3, 3))
    return tf.keras.models.Model(encoder_input,encoder_output,name='encoder_fftradon')

def ssim_loss(y_true, y_pred):
    """Structural simetry loss"""
    return 1-tf.reduce_mean(tf.image.ssim(y_true,y_pred, 1.0))

def test_loss(y_true, y_pred):
    """ Account for MSE, z derivative, gradient and l1 """
    alpha = 0.02                    #0.02   MSE z derivative
    beta  = 0.1                     #0.1    Blocky inversion (continuity ypred[i+1]-ypred[i]) #TODO increase beta
    l1 = 0.05                       #0.05   l1 norm
    v_max = 0.2                     #0.2    max value
    fact1 = 1-alpha-beta-l1-v_max   # MSE
    losses_f = []

    'Mean Square error'
    loss = tf.reduce_sum((y_true-y_pred)**2,axis=[1,2])/tf.reduce_sum(y_true ** 2, axis=[1, 2])
    loss = tf.reduce_mean(loss)
    losses_f.append(fact1 * loss)
    'Calculate mean squared error of the z derivative'
    if alpha !=0:
        dlabel = y_true[:,1:,:] - y_true[:,:-1,:]
        dout   = y_pred[:,1:,:] - y_pred[:,:-1,:]
        num = tf.reduce_sum((dlabel - dout)**2,axis=[1, 2])
        den = tf.reduce_sum(dlabel ** 2,axis=[1, 2]) + 1E-6
        loss = tf.reduce_mean(num / den)
        losses_f.append(alpha * loss)
    'Minimize gradient (blocky inversion)'
    if beta !=0:
        num = tf.reduce_sum(tf.abs(y_pred[:,1:,:] - y_pred[:,:-1,:]),axis=[1,2])
        den = tf.norm(y_pred, ord=1, axis=[1, 2]) / .02
        loss = tf.reduce_mean(num / den)
        losses_f.append(beta * loss)
    'l1 norm'
    if l1 !=0:
        loss = tf.reduce_sum(tf.abs(y_true-y_pred),axis=[1,2])/tf.reduce_sum(tf.abs(y_true),axis=[1,2])
        loss = tf.reduce_mean(loss)
        losses_f.append(l1 * loss)
    if v_max !=0:
        loss = tf.reduce_sum(tf.abs(tf.reduce_max(y_true,axis=1) - tf.reduce_max(y_pred,axis=1)),axis=-1)
        loss = tf.reduce_mean(loss)
        losses_f.append(v_max * loss)
    return tf.reduce_sum(losses_f)

class Weighted_loss(tf.keras.losses.Loss):
    """
    Calculate a weighted loss
    alpha: weight for the MSE z derivative
    beta: weight for the error in the blocky inversion
    l1: weight for the l1 norm error 
    v_max: weigth for the error in the maximum values
    fact1: =1-alpha-beta-l1-v_max, weigth for the MSE
    """
    def __init__(self,outlabel,alpha=0.02,beta=0.1,l1=0.05,v_max=0.2,weights='None'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.v_max = v_max
        if weights == 'None':
            weights = np.ones(len(outlabel))/len(outlabel)
        self.weights = weights
    # def __call__(self,y_true,y_pred):
    def call(self, y_true, y_pred):
        alpha = self.alpha  # 0.02   MSE z derivative
        beta = self.beta  # 0.1    Blocky inversion (continuity ypred[i+1]-ypred[i]) #TODO increase beta
        l1 = self.l1  # 0.05   l1 norm
        v_max = self.v_max  # 0.2    max value
        fact1 = 1 - alpha - beta - l1 - v_max  # MSE
        losses_f = []

        'Mean Square error'
        loss = tf.reduce_sum((y_true - y_pred) ** 2, axis=[1]) / tf.reduce_sum(y_true ** 2, axis=[1])
        losses_f.append(fact1 * loss)
        'Calculate mean squared error of the z derivative'
        if alpha != 0:
            dlabel = y_true[:, 1:, :] - y_true[:, :-1, :]
            dout = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            num = tf.reduce_sum((dlabel - dout) ** 2, axis=[1])
            den = tf.reduce_sum(dlabel ** 2, axis=[1]) + 1E-6
            loss = num/den
            losses_f.append(alpha * loss)
        'Minimize gradient (blocky inversion)'
        if beta != 0:
            num = tf.reduce_sum(tf.abs(y_pred[:, 1:, :] - y_pred[:, :-1, :]), axis=[1])
            den = tf.reduce_sum(tf.abs(y_pred), axis=[1]) / .02
            loss = num / den
            losses_f.append(beta * loss)
        'l1 norm'
        if l1 != 0:
            loss = tf.reduce_sum(tf.abs(y_true - y_pred), axis=[1]) / tf.reduce_sum(tf.abs(y_true), axis=[1])
            losses_f.append(l1 * loss)
        if v_max != 0:
            loss = tf.abs(tf.reduce_max(y_true, axis=1) - tf.reduce_max(y_pred, axis=1))
            losses_f.append(v_max * loss)
        return tf.reduce_sum(tf.reduce_sum(losses_f,axis=[0,1])*np.array(self.weights))

def nn_architecture(datatype:list, encoders_list:dict, output_depth:str, outlabel:list = 'vpvs',
                    blocky=False, categ=False, mult_outputs=False):
    """
    Generates an encoder decoder architecture for single to multiple inputs, and sigle or double output.
    Parameters
    ----------
    datatype : List[str]: choose between ['shotgather','dispersion','radon','fft_radon']
    encoders_list : dict: dictionary with encoders eg {'dispersion':encoder_disp,'shotgather':encoder_sg,
    'fft_radon':encoder_fftradon, 'radon':encoder_radon}
    output_depth : str: either '700m' or '1000m'
    outlabel : str: chose between 'vpvs','vp', or 'vs'
    blocky: True for calling the blocking decoder

    Returns
    -------
    cnn_model
    """

    encoders = {}
    for l in datatype:
        encoders[l] = encoders_list[l]()
        print(encoders[l].summary())

    metrics = []
    if blocky:
        decoder = decoder_700m_blocky(encoders[datatype[0]].output,output_depth=output_depth,ol=outlabel)
        loss = test_loss
        # loss = Weighted_loss(outlabel=outlabel[0:])
    elif categ:
        decoder = decoder_700m_cat(encoders[datatype[0]].output, output_depth=output_depth,ol=outlabel)
        loss = tf.keras.losses.categorical_crossentropy
        # loss = tf.keras.losses.MeanSquaredError
    else:
        decoder = decoder_700m(encoders[datatype[0]].output,output_depth=output_depth,ol=outlabel)
        loss = test_loss
        # loss = Weighted_loss(outlabel=outlabel[0:1])
    print(decoder.summary())

    if not mult_outputs:
        if len(datatype) != 1:
            combined = tf.concat([encoders[l].output for l in datatype], axis=-1)
            encoded_img = tf.keras.layers.Conv2D(filters=encoders[datatype[0]].output.shape[-1], kernel_size=(1, 1),
                                                 activation=tf.nn.leaky_relu, padding='same', strides=(1, 1))(combined)
            decoded_img = decoder(encoded_img)
            cnn_model = tf.keras.models.Model(inputs=[encoders[l].input for l in datatype], outputs=[decoded_img])
        else:
            encoded_img = encoders[datatype[0]].output
            decoded_img = decoder(encoded_img)
            cnn_model = tf.keras.models.Model(encoders[datatype[0]].input, decoded_img)
        cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss)
    else:
        combined = tf.concat([encoders[l].output for l in datatype], axis=-1)
        encoded_img = tf.keras.layers.Conv2D(filters=encoders[datatype[0]].output.shape[-1], kernel_size=(1, 1),
                                             activation=tf.nn.leaky_relu, padding='same', strides=(1, 1))(combined)
        decoded_img = [decoder_700m(encoders[datatype[0]].output,output_depth=output_depth,ol=[ol],name=ol)(encoded_img)
                        for ol in outlabel]
        cnn_model = tf.keras.models.Model(inputs=[encoders[l].input for l in datatype], outputs=decoded_img)
        # loss = Weighted_loss(outlabel=outlabel[0:1])
        # loss_weights = {ol:1 for ol in outlabel}
        if '1/q' in outlabel:
            loss_weights = {ol:(1-.02)/(len(outlabel)-1) for ol in outlabel if ol !='1/q'}
            loss_weights['1/q'] = 0.02
        else:
            loss_weights = {ol: 1/len(outlabel) for ol in outlabel}

        cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss,loss_weights=loss_weights)
    print(cnn_model.summary())
    return cnn_model

# %% Additional functions
def bandpass(data,dt,flow,fup):
    """ Bandpass filter for the shot gathers """
    temp = np.fft.fft(np.pad(data, [(500, 500), (500, 500)]), axis=0)
    freq = np.fft.fftfreq(np.shape(temp)[0],dt)
    filter_bp = np.zeros(np.shape(temp)[0])
    mask = (np.abs(freq) >= flow) & (np.abs(freq) <=fup)
    filter_bp[mask] = 1
    tukey = signal.tukey(200,0.5)
    filter_bp = np.reshape(signal.convolve(filter_bp,tukey,mode='same'),[-1,1])
    # filter_bp /= filter_bp.max()
    filter_bp /= np.max(filter_bp)
    temp *= filter_bp
    data_bp = np.fft.ifft(temp,axis=0)[500:-500,500:-500]
    return data_bp.real

def conv3d2d(data, dt, offset, method="singlevelocity", vphase=None, off1=None, off2=None, delay=0):
    """Code from Gabriel. 3D to 2D compensation"""
    nt = data.shape[0]
    no = data.shape[1]
    offset = np.reshape(np.abs(offset), [1, no])
    d = np.fft.fft(data, axis=0)
    w = np.reshape(2.0 * np.pi * np.fft.fftfreq(nt, dt), [nt, 1])
    ft = np.sqrt(np.pi / 2.0 / (np.abs(w))) * (1.0 - 1.0j * np.sign(w))
    ft = np.where(np.abs(w) > 0, ft, 0)
    d = np.real(np.fft.ifft(ft * d, axis=0))
    t = np.reshape(np.arange(0, nt) * dt - delay, (nt, 1))
    ti = np.where(t > 0, 1.0 / np.sqrt(t), 0)
    if method == "singlevelocity":
        famp = np.sqrt(2.0 * offset * vphase)
    elif method == "directwave":
        famp = np.sqrt(2.0) * offset * ti
    elif method == "hybrid":
        coef = (offset - off1) / (off2 - off1)
        coef = np.where(coef < 0, 0, coef)
        coef = np.where(coef > 1, 1, coef)
        coef = cosine_taper(coef, 1.0)
        famp = coef * np.sqrt(2.0) * offset * ti
        famp += (1.0 - coef) * np.sqrt(2.0 * offset * vphase)
    return famp * d
    
def random_noise(data, max_amp):
    """
    Add gaussian random noise to the data.

    :param data: The data array.
    :param max_amp: Maximum amplitude of the noise relative to the data max.

    :return: The data with noise.
    """
    max_amp = max_amp * np.max(data) * 2.0
    data = data + (np.random.rand(data.shape[0], data.shape[1])-.5)*max_amp
    return data

# %% Handling dataset
datafile = "/data/shared/seismic_data/GSC_Permafrost/2014/line06/line06_geom.sgy"
def read_shots(ffids, ng=120, ffid0=7469, file=datafile):
    """ Reading shots """
    segy = _read_segy(file, headonly=True)
    traces = segy.traces
    data = []
    for i, ffid in enumerate(ffids):
        n1 = (ffid - ffid0) * ng
        n2 = n1 + ng
        traces = segy.traces[n1:n2]
        temp = np.stack([trace.data for trace in traces
                         if trace.header.original_field_record_number == ffid])
        if temp.size == 0:
            raise ValueError("Could not find shot %d" % ffid)
        data.append(temp.T)
    data = np.array(data)
    dt = traces[0].header.sample_interval_in_ms_for_this_trace
    del segy
    return data, dt

def read_cmps(ng=120,file="/data/jbustamante/datapreprocessing/line06_cmds.sgy",bp=False,flow=15,fup=65):
    segy = _read_segy(file, headonly=True)
    traces = segy.traces
    data = np.stack([trace.data for trace in traces])
    dt = traces[0].header.sample_interval_in_ms_for_this_trace
    if bp:
        data = bandpass(data.T,dt/10**6,flow,fup).T
    data_out = []
    for i in range(data.shape[0]//ng):
        data_out.append(data[i*120:(i+1)*120,:].T)
    # dt = traces[0].header.sample_interval_in_ms_for_this_trace
    data_out = np.array(data_out)
    del segy
    return data_out,dt


def get_dataset(case,datatype,trainsize=2000,testsize=5):
    dataset_module = import_module("DefinedDataset." + case)
    dataset = getattr(dataset_module, case)()
    dataset.trainsize = trainsize
    dataset.validatesize = (dataset.trainsize // 10)
    dataset.testsize = testsize
    sigma = 3  # sigma value to apply in the gaussian smoothing for the labels

    dataset.generate_dataset(ngpu=1)
    dataset._getfilelist(phase='train')
    inputs = {}
    sizes_inp = {}

    'Defining sizes'
    for i in datatype:
        sizes_inp[i] = [*dataset.generator.read(dataset.files['train'][0])[0][i].shape, 1]
        inputs[i] = np.empty([dataset.trainsize, *sizes_inp[i]])
        inputs[i + '_test'] = np.empty([dataset.testsize, *sizes_inp[i]])
        inputs[i + '_validate'] = np.empty([dataset.validatesize, *sizes_inp[i]])
    sizes_lab = dataset.generator.read(dataset.files['train'][0])[1]['vpdepth'].shape[0]
    labels = np.empty([dataset.trainsize, 2, sizes_lab])
    labels_test = np.empty([dataset.testsize, 2, sizes_lab])
    labels_validate = np.empty([dataset.validatesize, 2, sizes_lab])

    'Getting train dataset'
    for i in range(dataset.trainsize):
        data = dataset.get_example()
        for j in datatype:
            if j == 'dispersion':
                inputs[j][i] = np.abs(data[0][j].reshape(sizes_inp[j]))
            if j == 'shotgather':
                data0 = data[0][j].reshape(-1, 120)
                inputs[j][i] = (data0.reshape(sizes_inp[j]))
        labels[i][0] = gaussian_filter(data[1]['vpdepth'], sigma=sigma).reshape(sizes_lab)
        labels[i][1] = gaussian_filter(data[1]['vsdepth'], sigma=sigma).reshape(sizes_lab)
    labels = np.transpose(labels, (0, 2, 1))

    'Getting test dataset'
    for i in range(dataset.testsize):
        data = dataset.get_example(phase='test')
        for j in datatype:
            if j == 'dispersion':
                inputs[j + '_test'][i] = np.abs(data[0][j].reshape(sizes_inp[j]))
            if j == 'shotgather':
                data0 = data[0][j].reshape(-1, 120)
                inputs[j + '_test'][i] = (data0.reshape(sizes_inp[j]))
        labels_test[i][0] = gaussian_filter(data[1]['vpdepth'], sigma=sigma).reshape(sizes_lab)
        labels_test[i][1] = gaussian_filter(data[1]['vsdepth'], sigma=sigma).reshape(sizes_lab)
    labels_test = np.transpose(labels_test, (0, 2, 1))

    'Getting validate dataset'
    for i in range(dataset.validatesize):
        data = dataset.get_example()
        for j in datatype:
            if j == 'dispersion':
                inputs[j + '_validate'][i] = np.abs(data[0][j].reshape(sizes_inp[j]))
            if j == 'shotgather':
                data0 = data[0][j].reshape(-1, 120)
                inputs[j + '_validate'][i] = (data0.reshape(sizes_inp[j]))
        labels_validate[i][0] = gaussian_filter(data[1]['vpdepth'], sigma=sigma).reshape(sizes_lab)
        labels_validate[i][1] = gaussian_filter(data[1]['vsdepth'], sigma=sigma).reshape(sizes_lab)
    labels_validate = np.transpose(labels_validate, (0, 2, 1))

    return inputs, labels, labels_test, labels_validate

def get_dataset2(case,datatype=['shotgather','dispersion','radon','fft_radon'],trainsize=2000,testsize=5):
    """
    Generates dataset for training, validation and testing to be used as input for a NN. The validation is of size
    trainsize//10. The function reads/generate shotgathers as defined by Deep2D_velocity code and transforms those shots
    to dispersion, radon or fft_radon. Note that the fft_radon can only be generated if radon is defined.
    Parameters:
    ----------
    case : str: case name as defined in the DefinedDataset folder
    datatype : choose between ['shotgater','dispersion','radon','fft_radon'] or a combination of elements
    trainsize : Number of samples in the training dataset
    testsize : Number of samples in the test dataset

    Returns
    ----------
    inputs: Dictionary with the datasets defined in datatype. 'shotgather' datatype will be always included
    labels, labels_test, labels_validate

    """
    if 'shotgather' not in datatype:
        print('\'shotgather\' must be included in datatype \n \tIncluding \'shotgather\' in datatype')
        datatype.append('shotgather')
    if 'fft_radon' in datatype and 'radon' not in datatype:
        print('\'radon\' datatype should be defined to create fft_radon \n \tIncluding \'radon\' in datatype')
        datatype.append('radon')

    dataset_module = import_module("DefinedDataset." + case)
    dataset = getattr(dataset_module, case)()
    dataset.trainsize = trainsize
    dataset.validatesize = (dataset.trainsize // 10)
    dataset.testsize = testsize
    sigma = 3
    fmax = 100
    dt = dataset.acquire.dt*dataset.acquire.resampling
    nt = dataset.acquire.NT//dataset.acquire.resampling
    t = np.arange(0, nt*dt, dt)

    off0 = np.abs(dataset.generator.seismic.rec_pos_all[0,0]-dataset.generator.seismic.src_pos_all[0,0])
    off1 = np.abs(dataset.generator.seismic.rec_pos_all[0,-1]-dataset.generator.seismic.src_pos_all[0,0])
    dg = np.abs(dataset.generator.seismic.rec_pos_all[0,0]-dataset.generator.seismic.rec_pos_all[0,1])
    ng = dataset.generator.seismic.rec_pos_all.shape[-1]
    offmin,offmax = np.min([off0,off1]), np.max([off0,off1])
    x = np.arange(offmin, offmin + dg*ng, 12.5)
    c = np.linspace(1000, 4500, 200)
    c_radon = np.linspace(1000, 3000, 200)

    dataset.generate_dataset(ngpu=1)
    dataset._getfilelist(phase='train')
    inputs = dict()
    sizes_inp = dict()

    'Defining sizes'
    shotg = np.float64(dataset.generator.read(dataset.files['train'][0])[0]['shotgather'])
    disp = dispersion(shotg.T,dt,x, c,fmax=fmax).numpy().T
    radon = hyperbolic_radon(shotg.T, t, x, c_radon).numpy().T
    freq = np.fft.fftfreq(radon.shape[0], dt)
    mask = (freq >= 0) & (freq < fmax)
    fft_radon = np.abs(np.fft.fft(radon, axis=0))[mask,:]
    sizes_inp['shotgather'] = [*shotg.shape,1]
    sizes_inp['dispersion'] = [*disp.shape,1]
    sizes_inp['radon'] = [*radon.shape,1]
    sizes_inp['fft_radon'] = [*fft_radon.shape,1]

    for i in datatype:
        inputs[i] = np.empty([dataset.trainsize, *sizes_inp[i]])
        inputs[i + '_test'] = np.empty([dataset.testsize, *sizes_inp[i]])
        inputs[i + '_validate'] = np.empty([dataset.validatesize, *sizes_inp[i]])
    sizes_lab = dataset.generator.read(dataset.files['train'][0])[1]['vpdepth'].shape[0]
    labels = np.empty([dataset.trainsize, 2, sizes_lab])
    labels_test = np.empty([dataset.testsize, 2, sizes_lab])
    labels_validate = np.empty([dataset.validatesize, 2, sizes_lab])

    'Getting train dataset'
    for i in tqdm(range(dataset.trainsize),desc= 'Getting train dataset'):
        data = dataset.get_example()
        d = data[0]['shotgather'].reshape(sizes_inp['shotgather'])
        inputs['shotgather'][i] = d/np.max(np.abs(d))
        d = data[0]['shotgather'].reshape((-1, 120))
        if 'dispersion' in datatype:
            disp = dispersion(d.T, dt, x, c, fmax=100).numpy().T
            inputs['dispersion'][i,:,:,0] = (disp-np.min(disp))/(np.max(disp)-np.min(disp))
        if 'radon' in datatype:
            radon = hyperbolic_radon(d.T, t, x, c_radon).numpy().T
            inputs['radon'][i, :, :, 0] = radon/np.max(np.abs(radon))
            if 'fft_radon' in datatype:
                fft_radon = np.abs(np.fft.fft(radon, axis=0))[mask, :]
                inputs['fft_radon'][i,:,:,0] = (fft_radon-np.min(fft_radon))/(np.max(fft_radon)-np.min(fft_radon))
        labels[i][0] = gaussian_filter(data[1]['vpdepth'], sigma=sigma).reshape(sizes_lab)
        labels[i][1] = gaussian_filter(data[1]['vsdepth'], sigma=sigma).reshape(sizes_lab)
    labels = np.transpose(labels, (0, 2, 1))

    'Getting validate dataset'
    for i in tqdm(range(dataset.validatesize),desc= 'Getting validate dataset'):
        data = dataset.get_example()
        d = data[0]['shotgather'].reshape(sizes_inp['shotgather'])
        inputs['shotgather_validate'][i] = d / np.max(np.abs(d))
        d = data[0]['shotgather'].reshape((-1, 120))
        if 'dispersion' in datatype:
            disp = dispersion(d.T, dt, x, c, fmax=100).numpy().T
            inputs['dispersion_validate'][i,:,:,0] = (disp - np.min(disp)) / (np.max(disp) - np.min(disp))
        if 'radon' in datatype:
            radon = hyperbolic_radon(d.T, t, x, c_radon).numpy().T
            inputs['radon_validate'][i, :, :, 0] = radon/np.max(np.abs(radon))
            if 'fft_radon' in datatype:
                fft_radon = np.abs(np.fft.fft(radon, axis=0))[mask, :]
                inputs['fft_radon_validate'][i, :, :, 0] = (fft_radon-np.min(fft_radon))/\
                                                           (np.max(fft_radon)-np.min(fft_radon))
        labels_validate[i][0] = gaussian_filter(data[1]['vpdepth'], sigma=sigma).reshape(sizes_lab)
        labels_validate[i][1] = gaussian_filter(data[1]['vsdepth'], sigma=sigma).reshape(sizes_lab)
    labels_validate = np.transpose(labels_validate, (0, 2, 1))

    'Getting test dataset'
    for i in tqdm(range(dataset.testsize),desc= 'Getting test dataset'):
        data = dataset.get_example()
        d = data[0]['shotgather'].reshape(sizes_inp['shotgather'])
        inputs['shotgather_test'][i] = d/np.max(np.abs(d))
        d = data[0]['shotgather'].reshape((-1, 120))
        if 'dispersion' in datatype:
            disp = dispersion(d.T, dt, x, c, fmax=100).numpy().T
            inputs['dispersion_test'][i,:,:,0] = (disp-np.min(disp)) / (np.max(disp)-np.min(disp))
        if 'radon' in datatype:
            radon = hyperbolic_radon(d.T, t, x, c_radon).numpy().T
            inputs['radon_test'][i, :, :, 0] = radon/np.max(np.abs(radon))
            if 'fft_radon' in datatype:
                fft_radon = np.abs(np.fft.fft(radon, axis=0))[mask, :]
                inputs['fft_radon_test'][i, :, :, 0] = (fft_radon-np.min(fft_radon))/\
                                                       (np.max(fft_radon)-np.min(fft_radon))
        labels_test[i][0] = gaussian_filter(data[1]['vpdepth'], sigma=sigma).reshape(sizes_lab)
        labels_test[i][1] = gaussian_filter(data[1]['vsdepth'], sigma=sigma).reshape(sizes_lab)
    labels_test = np.transpose(labels_test, (0, 2, 1))

    return inputs, labels, labels_test, labels_validate

def get_dataset_file(case,trainsize,testsize,datatype,create_noisy_file=False):
    """
    This function either read the inputs values of the NN from a file (noisy_data.mat) or call get_dataset2 function to
    create the inputs and the labels.
    and return the function
    Parameters
    ----------
    case : str: case name as defined in the DefinedDataset folder
    trainsize : int: Number of samples in the training dataset
    testsize : int: Number of samples in the test dataset
    create_noisy_file : bool: True to create the dataset, otherwise it will be readen from a file

    Returns
    -------
    inputs: Dictionary with the datasets defined in datatype. 'shotgather' datatype will be always included
    labels, labels_test, labels_validate

    """
    noisy_data_file = 'Datasets/noisy_data.mat'
    cases_list = []
    if not os.path.isfile(noisy_data_file): create_noisy_file = True
    else:
        noisy_file = h5.File(noisy_data_file, 'r')
        cases_list = list(noisy_file.keys())
        noisy_file.close()
        if case not in cases_list: create_noisy_file = True
    if create_noisy_file:
        if case in cases_list:
            noisy_file = h5.File(noisy_data_file, 'a')
            del noisy_file[case]
            noisy_file.close()
        inputs, labels, labels_test, labels_validate = get_dataset2(case, trainsize=trainsize, testsize=testsize)
        noisy_file = h5.File(noisy_data_file, 'a')
        [noisy_file.create_dataset(case + '/inputs/'+i, data=inputs[i]) for i in list(inputs)]
        noisy_file.create_dataset(case + '/labels', data=labels)
        noisy_file.create_dataset(case + '/labels_test', data=labels_test)
        noisy_file.create_dataset(case + '/labels_validate', data=labels_validate)
        noisy_file.close()
    else:
        noisy_file = h5.File(noisy_data_file, 'r')
        labels = noisy_file.get(case + '/labels')[()]
        labels_test = noisy_file.get(case + '/labels_test')[()]
        labels_validate = noisy_file.get(case + '/labels_validate')[()]
        inputs_temp = noisy_file.get(case + '/inputs/')
        inputs = dict()
        for i in list(inputs_temp):
            inputs[i] = inputs_temp[i][()]
        del inputs_temp
        noisy_file.close()
    inputs_out = dict()
    for i in datatype:
        inputs_out[i] = inputs[i]
        inputs_out[i+'_test'] = inputs[i+'_test']
        inputs_out[i + '_validate'] = inputs[i + '_validate']
    return inputs_out, labels, labels_test, labels_validate

def get_dataset_rec2(datatype,offset,create_file=False,filein=datafile,fileout='data_rec.mat',gather='shot',
                     bp=False,flow=15,fup=65):
    """
    Parameters
    ----------

    datatype : list[str] : A list of strings defining the datatype. Choose between
    ['shotgather','dispersion','radon','fft_radon']
    offset : vector with distance between source and receivers
    create_file : bool: If True re-write the information in the output file for the selected datatype
    filein : str: SEGY file where the shot gathers are contained
    fileout : str: .mat file where the different datatypes will be stored
    gather : str: either 'shot' or 'cmp'
    bp : True for applying bandpass to the recorded Shot Gather
    flow : Low limit frequency for the bandpass filter
    fup : Upper limit frequency fot the bandpass filter

    Returns
    -------
    data_rec: A dictionary with the dataset for the elements defined in datatype
    """
    datatypes = copy.copy(datatype)
    if 'fft_radon' in datatypes and 'radon' not in datatypes:
        datatypes.append('radon')
    datatype_list = []
    shot_numbers = np.arange(7469,8275)
    tmax,fmax = 2,100
    c = np.linspace(1000, 4500, 200)
    c_radon = np.linspace(1000,3000,200)
    if not os.path.isfile(fileout):
        create_file = True
    else:
        h5file = h5.File(fileout, 'r')
        datatype_list = list(h5file.keys())
        h5file.close()

    data_rec = dict()
    if create_file:
        if gather == 'shot':
            data, dt = read_shots(shot_numbers,file=filein)
        else:
            data, dt = read_cmps(file=filein,bp=bp)
        dt /= 10**6
        nt = np.int(tmax / dt)
        t = np.arange(0, nt*dt, dt)
        data = data[:, :nt, :]
        data = np.float64(np.array([data[j, :, :] / np.abs(data[j, :, :]).max() for j in range(data.shape[0])]))
        sizes = dict()

        # if bp:
        #     data = np.array([bandpass(data[i, :, :], dt, flow, fup) for i in range(data.shape[0])])

        for j in datatypes:
            if j in datatype_list:
                h5file = h5.File(fileout, 'a')
                del h5file[j]
                h5file.close()
        if 'shotgather' in datatypes:
            data_rec['shotgather'] = np.expand_dims(data,axis=-1)
        if 'dispersion' in datatypes:
            disp = dispersion(data[0,:,:].T, dt, offset, c, fmax=fmax).numpy().T
            sizes['dispersion'] = [*disp.shape,1]
            # data_rec['dispersion'] = np.empty([len(shot_numbers),*sizes['dispersion']])
            data_rec['dispersion'] = np.empty([data.shape[0], *sizes['dispersion']])
            for j in tqdm(range(data.shape[0]),desc='Obtaining Dispersion Dataset'):
                disp = dispersion(data[j,:,:].T,dt,offset,c,fmax=fmax).numpy().T
                data_rec['dispersion'][j,:,:,0] = (disp-np.min(disp))/(np.max(disp)-np.min(disp))
        if 'radon' in datatypes:
            radon = hyperbolic_radon(data[0,:,:].T,t,offset,c_radon).numpy().T
            sizes['radon'] = [*radon.shape,1]
            # data_rec['radon'] = np.empty([len(shot_numbers), *sizes['radon']])
            data_rec['radon'] = np.empty([data.shape[0], *sizes['radon']])
            freq = np.fft.fftfreq(radon.shape[0], dt)
            mask = (freq >= 0) & (freq < fmax)
            if 'fft_radon' in datatypes:
                fft_radon = np.abs(np.fft.fft(radon, axis=0))[mask,:]
                sizes['fft'] = [*fft_radon.shape,1]
                # data_rec['fft_radon'] = np.empty([len(shot_numbers),*sizes['fft']])
                data_rec['fft_radon'] = np.empty([data.shape[0], *sizes['fft']])
            for j in tqdm(range(data.shape[0]),desc='Obtaining Radon Dataset'):
                radon = hyperbolic_radon(data[j,:,:].T, t, offset, c_radon).numpy().T
                data_rec['radon'][j,:,:,0] = radon/np.max(np.abs(radon))
                if 'fft_radon' in datatypes:
                    fft_radon = np.abs(np.fft.fft(radon, axis=0))[mask, :]
                    data_rec['fft_radon'][j,:,:,0] = (fft_radon-np.min(fft_radon))/(np.max(fft_radon)-np.min(fft_radon))
        h5file = h5.File(fileout,'w')
        [h5file.create_dataset(j, data=data_rec[j]) for j in list(data_rec)]
        h5file.close()
    else:
        h5file = h5.File(fileout,'r')
        for j in list(h5file):
            data_rec[j] = h5file.get(j)[()]
        h5file.close()
    return data_rec

def generate_noisy_files(case,trainsize,testsize):
    """ The noise is introduced through the preprocess funtion in the ShotGather class as defined by the Defined dataset
     file"""
    dataset_module = import_module("DefinedDataset." + case)
    dataset = getattr(dataset_module, case)()
    dataset.trainsize = trainsize
    dataset.validatesize = validatesize = (dataset.trainsize // 10)
    dataset.testsize = testsize
    sigma = 3
    fmax = 100
    dt = dataset.acquire.dt*dataset.acquire.resampling
    nt = dataset.acquire.NT//dataset.acquire.resampling
    t = np.arange(0, nt*dt, dt)

    off0 = np.abs(dataset.generator.seismic.rec_pos_all[0,0]-dataset.generator.seismic.src_pos_all[0,0])
    off1 = np.abs(dataset.generator.seismic.rec_pos_all[0,-1]-dataset.generator.seismic.src_pos_all[0,0])
    dg = np.abs(dataset.generator.seismic.rec_pos_all[0,0]-dataset.generator.seismic.rec_pos_all[0,1])
    ng = dataset.generator.seismic.rec_pos_all.shape[-1]
    offmin,offmax = np.min([off0,off1]), np.max([off0,off1])
    x = np.arange(offmin, offmin + dg*ng, 12.5)
    c = np.linspace(1000, 4500, 200)
    c_radon = np.linspace(1000, 3000, 200)

    dataset.generate_dataset(ngpu=1)
    dataset._getfilelist(phase='train')
    inputs = dict()
    sizes_lab = dataset.generator.read(dataset.files['train'][0])[1]['vpdepth'].shape[0]
    labels = np.empty([2, sizes_lab])

    'Getting train dataset'
    for phase,size in zip(['train','validate','test'],[trainsize,validatesize,testsize]):
        for i in tqdm(range(size), desc=case[-12:-5] + '--> Getting ' + phase + ' dataset'):
            data = dataset.get_example(phase=phase)
            d = data[0]['shotgather'].reshape(-1,120)
            inputs['shotgather'] = np.expand_dims(d/np.max(np.abs(d)),axis=-1)
            disp = dispersion(d.T, dt, x, c, fmax=fmax, epsilon=1e-6).numpy().T
            inputs['dispersion'] = np.expand_dims((disp-np.min(disp))/(np.max(disp)-np.min(disp)),axis=-1)
            radon = hyperbolic_radon(d.T, t, x, c_radon).numpy().T
            inputs['radon'] = np.expand_dims(radon/np.max(np.abs(radon)),axis=-1)
            if i == 0:
                freq = np.fft.fftfreq(radon.shape[0], dt)
                mask = (freq >= 0) & (freq < fmax)
            fft_radon = np.abs(np.fft.fft(radon, axis=0))[mask, :]
            inputs['fft_radon'] = np.expand_dims((fft_radon-np.min(fft_radon))/(np.max(fft_radon)-np.min(fft_radon)),
                                                 axis=-1)
            labels[0] = data[1]['vpdepth'].reshape(sizes_lab)
            labels[1] = data[1]['vsdepth'].reshape(sizes_lab)

            noisy_file_str = 'Datasets/noisy_data/'+case+'/'+phase+'/' + str(i) + '.mat'
            noisy_file = h5.File(noisy_file_str,'a')
            [noisy_file.create_dataset('inputs/' + inp, data=inputs[inp]) for inp in list(inputs)]
            noisy_file.create_dataset('labels', data=labels.T)
            noisy_file.close()

def generate_noisy_files2(case,trainsize,testsize):
    """Part of the noise is introduced through the preprocess funtion in the ShotGather class as defined by the Defined
    dataset file. After calling preprocess, the dispersion plot will be generated. Then some random noise will be
    included to both the shotgather and dispersion plot. Radon and fft_radon will be generated from the noised
    shotgather"""
    dataset_module = import_module("DefinedDataset." + case)
    dataset = getattr(dataset_module, case)()
    dataset.trainsize = trainsize
    dataset.validatesize = validatesize = (dataset.trainsize // 10)
    dataset.testsize = testsize
    sigma = 3
    fmax = 100
    dt = dataset.acquire.dt*dataset.acquire.resampling
    nt = dataset.acquire.NT//dataset.acquire.resampling
    t = np.arange(0, nt*dt, dt)

    off0 = np.abs(dataset.generator.seismic.rec_pos_all[0,0]-dataset.generator.seismic.src_pos_all[0,0])
    off1 = np.abs(dataset.generator.seismic.rec_pos_all[0,-1]-dataset.generator.seismic.src_pos_all[0,0])
    dg = np.abs(dataset.generator.seismic.rec_pos_all[0,0]-dataset.generator.seismic.rec_pos_all[0,1])
    ng = dataset.generator.seismic.rec_pos_all.shape[-1]
    offmin,offmax = np.min([off0,off1]), np.max([off0,off1])
    x = np.arange(offmin, offmin + dg*ng, 12.5)
    c = np.linspace(1000, 4500, 200)
    c_radon = np.linspace(1000, 3000, 200)

    print('Generating clean dataset....')
    dataset.generate_dataset(ngpu=1)
    dataset._getfilelist(phase='train')
    inputs = dict()
    sizes_lab = dataset.generator.read(dataset.files['train'][0])[1]['vpdepth'].shape[0]
    labels = np.empty([4, sizes_lab])

    lims = {'q': [dataset.model.properties['q'][0], dataset.model.properties['q'][1]]}
    lims['1/q'] = [1 / lims['q'][1], 1 / lims['q'][0]]

    'Getting train,validating and testing datasets'
    print('Generating noisy dataset....')
    for phase,size in zip(['train','validate','test'],[trainsize,validatesize,testsize]):
        for i in tqdm(range(size), desc=case[-14:-5] + '--> Getting ' + phase + ' dataset'):
            data = dataset.get_example(phase=phase)
            d = data[0]['shotgather'].reshape(-1,120)

            fnl = len('%s/Datasets/%s/%s/example_' %(os.getcwd(),case,phase)) #file name length
            if phase == 'train': file_num = int(data[-1][fnl:])
            elif phase == 'validate': file_num = int(data[-1][fnl:]) - (trainsize-1)
            else: file_num = int(data[-1][fnl:]) - (trainsize + validatesize-1)

            noisy_file_str = 'Datasets/noisy_data/' + case + '/' + phase + '/' + str(file_num) + '.mat'

            if not os.path.isfile(noisy_file_str) and file_num >=0:
                disp = dispersion(d.T, dt, x, c, fmax=fmax, epsilon=1e-6).numpy().T
                disp = random_noise(disp, .005)
                radon0 = hyperbolic_radon(d.T, t, x, c_radon).numpy().T # no-noise radon use as input for fft_radon
                inputs['dispersion'] = np.expand_dims((disp - np.min(disp)) / (np.max(disp) - np.min(disp)), axis=-1)
                d = random_noise(d, .005)
                inputs['shotgather'] = np.expand_dims(d/np.max(np.abs(d)),axis=-1)
                radon = hyperbolic_radon(d.T, t, x, c_radon).numpy().T
                inputs['radon'] = np.expand_dims(radon/np.max(np.abs(radon)),axis=-1)
                # if i == 0:
                freq = np.fft.fftfreq(radon.shape[0], dt)
                mask = (freq >= 0) & (freq < fmax)
                fft_radon = np.abs(np.fft.fft(radon0, axis=0))[mask, :]
                inputs['fft_radon'] = np.expand_dims((fft_radon-np.min(fft_radon))/(np.max(fft_radon)-np.min(fft_radon)),
                                                     axis=-1)
                labels[0] = data[1]['vpdepth'].reshape(sizes_lab)
                labels[1] = data[1]['vsdepth'].reshape(sizes_lab)
                labels[2] = data[1]['qdepth'].reshape(sizes_lab)
                q = labels[2]*(lims['q'][1]-lims['q'][0])+lims['q'][0]
                labels[3] = (1/q - lims['1/q'][0])/(lims['1/q'][1]-lims['1/q'][0])

                # noisy_file_str = 'Datasets/noisy_data/'+case+'/'+phase+'/' + str(i) + '.mat'
                noisy_file = h5.File(noisy_file_str,'a')
                [noisy_file.create_dataset('inputs/' + inp, data=inputs[inp]) for inp in list(inputs)]
                noisy_file.create_dataset('labels', data=labels.T)
                noisy_file.close()
            else: continue

def read_inp2(folder,datatype,batchsize,phase=None,shuffle=True,outlabel=['vp','vs','q'],depth_train=700,
              categorical=False,mult_outputs=False):
    label_indx ={'vp':0,'vs':1,'q':2,'1/q':3}
    indx = [label_indx[i] for i in outlabel]
    d_indx = int(depth_train//2.5)
    def read_inputs(file):
        h5file = h5.File(file,'r')
        inputs_temp = h5file.get('inputs/')
        data = {l : tf.convert_to_tensor(inputs_temp[l][()],dtype=tf.float32) for l in datatype}
        data['labels'] = tf.convert_to_tensor(h5file.get('labels')[()][:d_indx,indx],dtype=tf.float32)
        h5file.close()
        return tuple(data[l] for l in (datatype+['labels']))

    def read_inputs_mult(file):
        h5file = h5.File(file,'r')
        inputs_temp = h5file.get('inputs/')
        data = {l : tf.convert_to_tensor(inputs_temp[l][()],dtype=tf.float32) for l in datatype}
        temp_lab = h5file.get('labels')[()][:d_indx,indx]
        data.update({ol: tf.convert_to_tensor(temp_lab[:,i:i+1],dtype=tf.float32) for i,ol in enumerate(outlabel)})
        h5file.close()
        return tuple(data[l] for l in (datatype+outlabel))

    def read_inputs_cat(file):
        ncath = 100
        h5file = h5.File(file, 'r')
        inputs_temp = h5file.get('inputs/')
        data = {l: tf.convert_to_tensor(inputs_temp[l][()], dtype=tf.float32) for l in datatype}
        lab_r = h5file.get('labels')[()][:d_indx,indx]
        h5file.close()
        labels = np.zeros((*lab_r.shape,ncath))
        for i in range(ncath):
            mask = (lab_r >= i / 100) & (lab_r < (i + 1) / 100)
            labels[mask, i] = 1
        data['labels'] = tf.convert_to_tensor(labels,dtype=tf.float32)
        return tuple(data[l] for l in (datatype + ['labels']))

    def read_inputs2(file):
        if mult_outputs:
            tout = ([tf.float32 for l in datatype+outlabel])
            data = tf.numpy_function(read_inputs_mult, inp=[file], Tout=tout)
            inputs = {datat: data[i] for i, datat in enumerate(datatype)}
            labels = {lab: data[len(datatype)+i] for i,lab in enumerate(outlabel)}
        else:
            tout = [tf.float32 for l in datatype]
            tout.append(tf.float32)
            if categorical:
                data = tf.numpy_function(read_inputs_cat, inp=[file], Tout=tout)
            else:
                data = tf.numpy_function(read_inputs, inp=[file],Tout=tout)
            inputs = {datat : data[i] for i,datat in enumerate(datatype)}
            labels = data[-1]
        return inputs, labels

    if phase == 'test': shuffle = False
    for i in range(len(folder)):
        files = folder[i] + '*.mat'
        temp = tf.data.Dataset.list_files(files,shuffle=shuffle)
        if i ==0 :
            dataset = copy.copy(temp)
        else:
            dataset = dataset.concatenate(temp)

    dataset = dataset.map(read_inputs2,
                          num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=batchsize) #.prefetch(tf.data.AUTOTUNE)
    return dataset