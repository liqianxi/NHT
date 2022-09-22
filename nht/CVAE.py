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

from nht.MLP import MLP
from nht.utils import kl_divergence


class CVAE(keras.Model):
    def __init__(self, input_dim, latent_dim, cond_dim, lip_coeff=None, 
                        step_size = 0.0001, beta = 0.001, hiddens=[256,256], activation='tanh'):
        super(CVAE, self).__init__()

        self.beta = beta
        self.target_sigma = 1.0
        self.L = None
        self.anneal = None
        self.step_size = step_size

        self.steps = 0

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.lip_coeff = lip_coeff

        self.encoder = MLP(input_dim+cond_dim,hiddens,latent_dim*2,activation)
        self.decoder = MLP(latent_dim+cond_dim,hiddens,input_dim,activation)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.step_size)
        
        self.train_recon_loss = tf.contrib.eager.metrics.Mean(name='train_recon_loss')
        self.val_recon_loss = tf.contrib.eager.metrics.Mean(name='val_recon_loss')
        self.train_kl_loss = tf.contrib.eager.metrics.Mean(name='train_kl_loss')
        self.val_kl_loss = tf.contrib.eager.metrics.Mean(name='val_kl_loss')
        self.train_loss = tf.contrib.eager.metrics.Mean(name='train_loss')
        self.val_loss = tf.contrib.eager.metrics.Mean(name='val_loss')

        print(f'\n\n\n step size: {self.step_size} \n\n\n')
        print(f'\n\n\n beta: {self.beta} \n\n\n')

    def encode(self, cond_inp, recon_inp):
        x = tf.concat((recon_inp, cond_inp),axis=-1)
        z_params = self.encoder(x)
        mu = z_params[:,:self.latent_dim]
        sigma = z_params[:,self.latent_dim:]
        return mu, tf.keras.activations.softplus(sigma)
    
    def decode(self, z, cond_inp):
        return self.decoder(tf.concat((z,cond_inp),axis=-1))

    def call(self, cond_inp, recon_inp):
        mu, sigma = self.encode(self, cond_inp, recon_inp)
        z = self.reparameterization(mu, sigma)
        x_hat = self.decode(z,cond_inp)
        return x_hat

    def reparameterization(self, mu, sigma):
        eps = tf.random.normal(sigma.shape)
        return mu + eps*sigma

    def calculate_elbo(self, x, x_hat, mu, sigma, beta, target_sigma):
        #recon_loss = tf.losses.mean_squared_error(x, x_hat)
        recon_loss = tf.math.reduce_mean(tf.norm(x-x_hat, axis=1))
        kl_loss = beta * kl_divergence(mu, sigma, target_sigma)

        loss = recon_loss + kl_loss

        return recon_loss, kl_loss, loss

    def weight_proj(self, W, lam):
        scale_factor = 1/tf.stop_gradient(tf.math.maximum(1,tf.norm(W,ord=2)/lam)) # from Gouk et al Reg of NN by Enforcing Lipschitz Continuity
        # if scale_factor < 1:
        #     print('reprojected')
        
        return scale_factor*W

    def project_weights(self, network):
        W_0 = network.get_weights().copy()
        for i, w in enumerate(W_0):
            if len(w.shape) > 1: # not bias weights
                projected_w = self.weight_proj(w, self.L)
                W_0[i] = projected_w
       
        network.set_weights(W_0)

    def train_step(self, cond_inp, qdot):
        with tf.GradientTape() as tape:
            mu, sigma = self.encode(cond_inp, qdot)
            z = self.reparameterization(mu, sigma)
            qdot_hat = self.decode(z,cond_inp)
            #beta = min(self.beta, min(self.steps,self.anneal)/self.anneal)
            beta = self.beta
            target_sigma = self.target_sigma
            recon_loss, kl_loss, loss = self.calculate_elbo(qdot, qdot_hat, mu, sigma, beta, target_sigma)
            
            

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if self.L is not None: # Lipschitz Regularization
            #self.project_weights(self.encoder.net)
            self.project_weights(self.decoder.net)


        self.steps += 1

        # update logs
        self.train_recon_loss(recon_loss)
        self.train_kl_loss(kl_loss)
        self.train_loss(loss)


    def val_step(self, cond_inp, qdot):
        mu, sigma = self.encode(cond_inp, qdot)
        z = self.reparameterization(mu, sigma)
        qdot_hat = self.decode(z,cond_inp)
        recon_loss, kl_loss, loss = self.calculate_elbo(qdot, qdot_hat, mu, sigma, self.beta, self.target_sigma)

        #update logs
        self.val_recon_loss(recon_loss)
        self.val_kl_loss(kl_loss)
        self.val_loss(loss)


    



