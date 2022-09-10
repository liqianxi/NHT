# disable deprecation and future warnings from tf1
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import copy

import json
import tensorflow as tf

from MLP import MLP


class CAE(keras.Model):
    def __init__(self, input_dim, latent_dim, cond_dim, lip_coeff, hiddens=[256,256]):
        super(CAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.lip_coeff = lip_coeff

        self.encoder = MLP(input_dim+cond_dim,hiddens,latent_dim,'tanh')
        self.decoder = MLP(latent_dim+cond_dim,hiddens,input_dim,'tanh')
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        
        self.train_loss = tf.contrib.eager.metrics.Mean(name='train_loss')
        self.val_loss = tf.contrib.eager.metrics.Mean(name='val_loss')
    
    def call(self, cond_inp, recon_inp):
        x = tf.concat((recon_inp, cond_inp),axis=-1)
        z = self.encoder(x)
        x_hat = self.decoder(tf.concat((z,cond_inp),axis=-1))
        return x_hat

    def weight_proj(self, W, lam):
        scale_factor = 1/tf.stop_gradient(tf.math.maximum(1,tf.norm(W,ord=2)/lam)) # from Gouk et al Reg of NN by Enforcing Lipschitz Continuity
        # if scale_factor > 1 or scale_factor < 1:
        #     print('reprojected')
        
        return scale_factor*W

    def train_step(self, cond_inp, qdot):
        with tf.GradientTape() as tape:
            predictions = self.call(cond_inp, qdot)
            loss = tf.losses.mean_squared_error(qdot, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update logs
        self.train_loss(loss)


    def val_step(self, cond_inp, qdot):
        predictions = self.call(cond_inp, qdot)
        loss = tf.losses.mean_squared_error(qdot, predictions)

        #update logs
        self.val_loss(loss)


    



