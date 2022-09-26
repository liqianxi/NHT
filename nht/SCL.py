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
from nht.utils import project_weights
import copy

import json
import tensorflow as tf
from nht.MLP import MLP


class SCL(keras.Model):
    def __init__(self, action_dim, output_dim, cond_dim, lip_coeff=1, step_size = 0.0001, hiddens=[256,256], activation='tanh', action_pred=False):
        super(SCL, self).__init__()

        self.L = lip_coeff
        self.step_size = step_size
        self.action_pred = action_pred
        
        self.action_dim = action_dim
        self.output_dim = output_dim
        
        layers = len(hiddens) + 1
        self.L_layer = np.power(self.L, 1/layers)

        self.h = MLP(cond_dim, hiddens, action_dim*output_dim, activation)
            
        if action_pred:
            self.f = MLP(output_dim+cond_dim,hiddens,action_dim, activation) # encoder, predicts low dim action given state and high dim action

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.step_size)
        
        self.train_loss = tf.contrib.eager.metrics.Mean(name='train_recon_loss')
        self.val_loss = tf.contrib.eager.metrics.Mean(name='val_recon_loss')
    
    def freeze_model(self):
        for layer in self.h.net.layers:
            layer.get_config()
            layer.trainable = False
        self.h.freeze()
    
    def _get_map(self, inputs):
        xi_bar = self.h(inputs)
        Q = tf.reshape(xi_bar, [-1, self.output_dim, self.action_dim])
        
        return Q
    
    def _get_best_action(self, Q, u):
        Q_dagger = tf.linalg.pinv(Q) # does not assume Q is orthonormal
        u = tf.expand_dims(u,-1)
        a_star = tf.matmul(Q_dagger, u)

        return a_star

    def call(self, c, u):
        Q = self._get_map(c)
        if self.action_pred: # action prediction
            a = self.predict_action(u, c)
            u_hat = tf.matmul(Q, tf.expand_dims(a,-1))
            
        else: # action projection
            a_star = self._get_best_action(Q, u)
            u_hat = tf.matmul(Q, a_star)

        return u_hat

    def train_step(self, c, u):
        with tf.GradientTape() as tape:
            u_hat = self.call(c, u)
            u = tf.expand_dims(u,-1)
            loss = tf.math.reduce_mean(tf.norm(u-u_hat, axis=1))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        if self.L is not None: # Lipschitz Regularization
            project_weights(self.h, self.L_layer)
        
        self.train_loss(loss)

    def val_step(self, c, u):
        u_hat = self.call(c, u)
        u = tf.expand_dims(u,-1)
        loss = tf.math.reduce_mean(tf.norm(u-u_hat, axis=1))

        self.val_loss(loss)

    def predict_action(self, u, c):
        return self.f(tf.concat((u, c),axis=-1))





