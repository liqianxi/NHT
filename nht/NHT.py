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



class NHT(keras.Model):
    def __init__(self, action_dim, output_dim, cond_dim, lip_coeff=1, step_size = 0.0001, hiddens=[256,256], activation='tanh', use_exp=True, action_pred=False):
        super(NHT, self).__init__()

        self.L = lip_coeff
        self.step_size = step_size
        self.action_pred = action_pred

        self.action_dim = action_dim
        self.output_dim = output_dim
        self.use_exp = use_exp
        
        # correct for 2*sqrt(k) Lipschitz coefficient of Householder tranform
        k = self.action_dim
        layers = len(hiddens) + 1
        self.L_NN = self.L/(2*np.sqrt(k))
        self.L_layer = np.power(self.L_NN, 1/layers)

        if self.use_exp:
            self.h = MLP(cond_dim, hiddens, action_dim*(output_dim-1), activation)
        else:
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

    def _exp_map(self,H):
        epsilon = 1e-6
        # H.shape: batch x n-1 x k
        n_minus_1 = int(H.shape[-2])
        zero_row = tf.zeros((1,n_minus_1))        # 1 x n-1
        I = tf.eye(n_minus_1)                     # n-1 x n-1
        zero_I = tf.concat((zero_row, I), axis=0) # n x n-1

        v_mat = tf.matmul(zero_I, H)                  # batch x n x k
        v_norm_vec = tf.linalg.norm(v_mat,axis=1)     # batch x k
        v_norm_row_vec = tf.expand_dims(v_norm_vec,1) # batch x 1 x k
        v_norm_diag = tf.linalg.diag(v_norm_vec)      # batch x k x k
        inv_v_norm_diag = tf.linalg.diag(1/(v_norm_vec+epsilon))
        sin_over_norm = tf.linalg.matmul(tf.math.sin(v_norm_diag),inv_v_norm_diag) # batch x k x k

        e1 = tf.eye(n_minus_1+1,1) # n x 1
        
        cos_term = tf.linalg.matmul(e1, tf.math.cos(v_norm_row_vec))
        sin_term = tf.linalg.matmul(v_mat, sin_over_norm)
        
        exp_mapped_vs = cos_term+sin_term
        
        return exp_mapped_vs
        

    
    def _householder(self, H):
        # Note, we assume the columns of H already have unit length, 
        # thanks to the exponential map to the unit sphere
        k = H.shape[-1] # number of vectors from which to construct reflections
        Q = tf.eye(int(H.shape[-2]))
        I = tf.eye(int(H.shape[-2]))
        
        for c in range(k):
            v = tf.expand_dims(H[:,:,c],-1) # batch x n x 1
            vT = tf.transpose(v,(0,2,1))    # batch x 1 x n
            vvT = tf.matmul(v,vT)           # batch x n x n
            H_i =  I - 2*vvT                # batch x n x n
            Q = tf.matmul(Q, H_i)   # batch x n x n

        return Q[:,:,:k] # H_hat


    def _get_map(self, inputs):
        x = self.h(inputs)
        if self.use_exp:
            v_bar = tf.reshape(x, [-1, self.output_dim-1, self.action_dim])
            v_hat_bar = self._exp_map(v_bar)
        else:
            v_hat_bar = tf.reshape(x, [-1, self.output_dim, self.action_dim])
            
        Q = self._householder(v_hat_bar)
        
        return Q
    
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
            project_weights(self.h, self.L_layer)
        
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



