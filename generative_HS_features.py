from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import argparse
import os
import glob

import tensorflow as tf 
from tensorflow import keras

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, UpSampling3D, Input, ZeroPadding3D, Lambda, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D
from keras.losses import mse, binary_crossentropy,mean_absolute_error 
from keras.utils import plot_model
from keras.constraints import unit_norm, max_norm
from keras import regularizers
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
import numpy as np
import nibabel as nib
import scipy as sp
import scipy.ndimage
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.manifold import MDS
import matplotlib.pyplot as plt




dropout_alpha = 
ft_bank_baseline = 
latent_dim = 
augment_size = 
L2_reg= 

def sampling(args):
    """Reparameterization trick 
    # Arguments:
        args (tensor): mean and log of variance sampled latent vector
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    thre = K.random_uniform(shape=(batch,1))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
    


# build encoder model
    
feature = Conv3D(16, activation='relu', kernel_size=(10, 10, 10),padding='same')(input_image)
feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)
    
    
    
feature = Conv3D(32, activation='relu', kernel_size=(5, 5, 5),padding='same')(feature)
feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)
    
       
feature = Conv3D(64, activation='relu', kernel_size=(3, 3, 3),padding='same')(feature)
feature = MaxPooling3D(pool_size=(2, 2, 2))(feature)
        
feature = Flatten()(feature)
feature = Dropout(dropout_alpha)(feature)
        
feature_dense = Dense(64, activation='tanh',kernel_regularizer=regularizers.l2(L2_reg))(feature)
    
feature_z_mean = Dense(32, activation='tanh')(feature_dense)
z_mean = Dense(16, name='z_mean')(feature_z_mean)
feature_z_log_var = Dense(32, activation='tanh')(feature_dense)
z_log_var = Dense(16, name='z_log_var')(feature_z_log_var)
    
feature_r_mean = Dense(32, activation='tanh')(feature_dense)
r_mean = Dense(1, name='r_mean')(feature_r_mean)
feature_r_log_var = Dense(32, activation='tanh')(feature_dense)
r_log_var = Dense(1, name='r_log_var')(feature_r_log_var)   
    
        # use reparameterization trick to push the sampling out as input
z = Lambda(sampling, output_shape=(16,), name='z')([z_mean, z_log_var])
r = Lambda(sampling, output_shape=(1,), name='r')([r_mean, r_log_var])
    
        # instantiate encoder model
encoder = Model([input_image,input_r], [z_mean, z_log_var, z, r_mean, r_log_var, r], name='encoder')
encoder.summary()

# build generator model
generator_input = Input(shape=(1,), name='genrator_input')
        #inter_z_1 = Dense(int(latent_dim/4), activation='tanh', kernel_constraint=unit_norm(), name='inter_z_1')(generator_input)
        #inter_z_2 = Dense(int(latent_dim/2), activation='tanh', kernel_constraint=unit_norm(), name='inter_z_2')(inter_z_1)
        #pz_mean = Dense(latent_dim, name='pz_mean')(inter_z_2)
pz_mean = Dense(latent_dim, name='pz_mean', kernel_constraint=unit_norm())(generator_input)
pz_log_var = Dense(1, name='pz_log_var',kernel_constraint=max_norm(0))(generator_input)
    
    # instantiate generator model
generator = Model(generator_input, [pz_mean,pz_log_var], name='generator')
generator.summary() 
    
#    # build decoder model
#    
latent_input = Input(shape=(16,), name='z_sampling')
decoded = Dense(16, activation='tanh',kernel_regularizer=regularizers.l2(L2_reg))(latent_input)
decoded = Dense(36, activation='tanh',kernel_regularizer=regularizers.l2(L2_reg))(decoded)
#decoded = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(L2_reg))(decoded)
decoded = Reshape((4,3,3,1))(decoded)
#        
decoded = Conv3D(4, padding='same', kernel_size=(3, 3, 3), activation='relu')(decoded)
decoded = UpSampling3D((3,3,3))(decoded)
#    
decoded = Conv3D(8, padding='same', kernel_size=(3, 3, 3), activation='relu')(decoded)
decoded = UpSampling3D((4,2,2))(decoded)
#    
decoded = Conv3D(16, padding='same', kernel_size=(3, 3, 3), activation='relu')(decoded)
decoded = UpSampling3D((2,1,1))(decoded)
#    
decoded = Conv3D(1, kernel_size=(3, 3, 3),padding='same')(decoded)
#    
outputs = decoded
#    
   # instantiate decoder model
decoder = Model(latent_input, outputs, name='decoder')
decoder.summary()
#    
    # instantiate VAE model
pz_mean,pz_log_var = generator(encoder([input_image,input_r])[5])
outputs = decoder(encoder([input_image,input_r])[2])
vae = Model([input_image,input_r], [outputs, pz_mean,pz_log_var], name='vae_mlp')
vae.summary()
