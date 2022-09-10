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
from scl.utils import project_weights
#tf.enable_eager_execution()
import copy

import json
import tensorflow as tf
from scl.MLP import MLP




class SCL(keras.Model):
    def __init__(self, action_dim, output_dim, cond_dim, lip_coeff=1, step_size = 0.0001, hiddens=[256,256], activation='tanh', action_pred=False):
        super(SCL, self).__init__()

        self.L = lip_coeff
        self.step_size = step_size
        self.action_pred = action_pred

        self.action_dim = action_dim
        self.output_dim = output_dim

        self.h = MLP(cond_dim, hiddens, action_dim*output_dim, activation)
        if action_pred:
            self.f = MLP(output_dim+cond_dim,hiddens,action_dim, activation) # encoder, predicts low dim action given state and high dim action

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.step_size)
        
        self.train_loss = tf.contrib.eager.metrics.Mean(name='train_recon_loss')
        self.val_loss = tf.contrib.eager.metrics.Mean(name='val_recon_loss')
    
    
    def _householder(self, H):
        n_cols = H.shape[-1] # number of vectors from which to construct reflections
        Q = tf.eye(int(H.shape[-2]))
        I = tf.eye(int(H.shape[-2]))
        
        for c in range(n_cols):
            v = tf.expand_dims(H[:,:,c],-1)
            vT = tf.transpose(v,(0,2,1))
            vvT = tf.matmul(v,vT) 
            vTv = tf.matmul(vT,v)
            Qi =  I - 2*vvT/vTv
            Q = tf.matmul(Qi, Q)
        return Q[:,:,:n_cols]


    def _get_map(self, inputs):
        x = self.h(inputs)
        SCL_map = tf.reshape(x, [-1, self.output_dim, self.action_dim])
        SCL_map = self._householder(SCL_map)
        
        return SCL_map
    
    def _get_best_action(self, SCL_map, low_level_action):
        SCL_map_pinv = tf.transpose(SCL_map,perm=[0,2,1]) # assumes SCL_map is orthonormal
        low_level_action = tf.expand_dims(low_level_action,-1)
        least_square_sol = tf.matmul(SCL_map_pinv,low_level_action)

        return least_square_sol


    def train_step(self, cond_inp, qdot):
        with tf.GradientTape() as tape:
            qdot_hat = self.call(cond_inp, qdot)
            qdot = tf.expand_dims(qdot,-1)
            loss = tf.math.reduce_mean(tf.norm(qdot-qdot_hat, axis=1))


        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        if self.L is not None: # Lipschitz Regularization
            #project_weights(self.f, self.L)
            project_weights(self.h, self.L)
        
        self.train_loss(loss)


    def val_step(self, cond_inp, qdot):
        qdot_hat = self.call(cond_inp, qdot)
        qdot = tf.expand_dims(qdot,-1)
        loss = tf.math.reduce_mean(tf.norm(qdot-qdot_hat, axis=1))

        self.val_loss(loss)

    def predict_action(self, low_level_action, cond_inp):
        return self.f(tf.concat((low_level_action, cond_inp),axis=-1))

    def call(self, cond_inp, low_level_action):
        SCL_map = self._get_map(cond_inp)
        if self.action_pred: # action prediction
            a = self.predict_action(low_level_action, cond_inp)
            q_dot_hat = tf.matmul(SCL_map, tf.expand_dims(a,-1))
            
        else: # action projection
            a_star = self._get_best_action(SCL_map, low_level_action)
            q_dot_hat = tf.matmul(SCL_map, a_star)

        return q_dot_hat

    def test_orthonormality(self, SCL_map):
        print(tf.norm(SCL_map,axis=1))
        print(SCL_map[0,:,0].shape)
        print(tf.tensordot(SCL_map[0,:,0],SCL_map[0,:,1],axes=1))
        #print(tf.tensordot(SCL_map[0,:,0],SCL_map[0,:,2],axes=1))
        #print(tf.tensordot(SCL_map[0,:,1],SCL_map[0,:,2],axes=1))
        #print(tf.tensordot(SCL_map[0,:,1],SCL_map[0,:,3],axes=1))


