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
from utils import col_replacement, get_dataset
#tf.enable_eager_execution()
import copy

import json
import tensorflow as tf

# Create a tf.keras model.
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(1, input_shape=[10]))
# model.summary()

# # Save the tf.keras model in the SavedModel format.
# path = '/tmp/simple_keras_model'
# #tf.keras.experimental.export_saved_model(model, path)

# # Load the saved keras model back.
# new_model = tf.keras.experimental.load_from_saved_model(path)
# new_model.summary()


# class CustomModel(layers.Layer):
#     def __init__(self, hidden_units):
#         super(CustomModel, self).__init__()
#         self.input_names = 'test'
#         self.hidden_units = hidden_units
#         self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

#     def call(self, inputs):
#         x = inputs
#         for layer in self.dense_layers:
#             x = layer(x)
#         return x

#     def get_config(self):
#         return {"hidden_units": self.hidden_units}

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


# # model = CustomModel([16, 16, 10])
# # # Build the model by calling it
# input_arr = tf.random.uniform((1, 5))
# # outputs = model(input_arr)

# inputs = keras.Input(shape=input_arr.shape)
# outputs = CustomModel([16, 16, 10])(inputs)
# model = keras.Model(inputs, outputs)
# keras.experimental.export_saved_model(model, "my_model", serving_only=True)

# # Option 1: Load with the custom_object argument.
# # loaded_1 = tf.compat.v1.keras.experimental.load_from_saved_model(
# #     "my_model", custom_objects={"CustomModel": CustomModel}
# # )
# loaded_1 = tf.compat.v1.keras.experimental.load_from_saved_model("my_model", custom_objects={"CustomModel": CustomModel})

# # Option 2: Load without the CustomModel class.

# # Delete the custom-defined model class to ensure that the loader does not have
# # access to it.
# # del CustomModel

# # loaded_2 = keras.models.load_model("my_model")
# # np.testing.assert_allclose(loaded_1(input_arr), outputs)
# # np.testing.assert_allclose(loaded_2(input_arr), outputs)

# print("Original model:", model)
# print("Model Loaded with custom objects:", loaded_1)
# print("Model loaded without the custom object class:", loaded_2)









class Linear(layers.Layer):
    def __init__(self, input, units=256):
        super(Linear, self).__init__()
        self.units = units
        self.w = self.add_weight(
            name = 'W',
            shape=(input, self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name = 'b',
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    # def build(self, input_shape):
    #     self.w = self.add_weight(
    #         shape=(input_shape[-1], self.units),
    #         initializer="random_normal",
    #         trainable=True,
    #     )
    #     self.b = self.add_weight(
    #         shape=(self.units,), initializer="random_normal", trainable=True
    #     )

    def get_config(self):
        return {"w": self.w, "b": self.b}

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b



# class SCL(layers.Layer):
#     def __init__(self, action_dim, output_dim):
#         super(SCL, self).__init__()
#         self.action_dim = action_dim
#         self.output_dim = output_dim

#         self.linear_1 = Linear(128)
#         self.linear_2 = Linear(128)
#         self.linear_3 = Linear(action_dim*output_dim)

    

#     def _proj_u_onto_v(self, u, v):
#         u_dot_v = tf.matmul(tf.expand_dims(u,1),tf.expand_dims(v,-1))
#         v_dot_v = tf.matmul(tf.expand_dims(v,1),tf.expand_dims(v,-1))
#         proj = u_dot_v/v_dot_v*tf.expand_dims(v,-1)
#         return proj

#     def _gram_schmidt(self, H):
#         n_cols = H.shape[-1]
#         for c in range(n_cols):
#             # make col orthogonal to previous columns
#             col = H[:,:,c]
#             for prev_c in range(c):
#                 projection = tf.squeeze(self._proj_u_onto_v(col,H[:,:,prev_c]))  # project col onto previous column
#                 col = col - projection

#             # scale column s.t. it has norm 1
#             norm_of_col = tf.expand_dims(tf.norm(col,axis=-1),-1)
#             e = col/norm_of_col
#             H = col_replacement(H,c,e)
#         return H

#     #@tf.function
#     def _get_map(self, inputs):
#         x = self.linear_1(inputs)
#         x = tf.nn.tanh(x)
#         x = self.linear_2(x)
#         x = tf.nn.tanh(x)
#         x = self.linear_3(x)
#         SCL_map = tf.reshape(x, [x.shape[0], self.output_dim, self.action_dim])
#         SCL_map = self._gram_schmidt(SCL_map)
#         return SCL_map
    
#     def _get_best_action(self, SCL_map, low_level_action):
#         SCL_map_pinv = tf.transpose(SCL_map,perm=[0,2,1]) # assumes SCL_map is orthonormal
#         low_level_action = tf.expand_dims(low_level_action,-1)
#         least_square_sol = tf.matmul(SCL_map_pinv,low_level_action)
#         return least_square_sol

#     def call(self, cond_inp, low_level_action):
#         SCL_map = self._get_map(cond_inp)
        
#         # troubleshooting print statements
#         #print(tf.norm(SCL_map,axis=1))
#         #print(SCL_map[0,:,0].shape)
#         #print(tf.tensordot(SCL_map[0,:,0],SCL_map[0,:,1],axes=1))

#         a_star = self._get_best_action(SCL_map, low_level_action)
#         lla_projection = tf.matmul(SCL_map, a_star)
#         return lla_projection


class SCL(keras.Model):
    def __init__(self, action_dim, output_dim, cond_dim, lip_coeff):
        super(SCL, self).__init__()

        self.action_dim = action_dim
        self.output_dim = output_dim
        self.lip_coeff = lip_coeff

        self.linear_1 = Linear(cond_dim, 256)
        self.linear_2 = Linear(256, 256)
        self.linear_3 = Linear(256, action_dim*output_dim)

        #self.mse = tf.keras.losses.MeanSquaredError()
        #self.mse = tf.compat.losses.MeanSquaredError()
        #self.mse = tf.losses.mean_squared_error()
        #self.optimizer = tf.keras.optimizers.Adam()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        print('initialing old version of SCL - you are somehow still using this')
        mask0 = np.ones((output_dim,action_dim))
        mask0[:,0] = 0
        mask0 = np.expand_dims(mask0,0)
        mask1 = np.ones((output_dim,action_dim))
        mask1[:,1] = 0
        mask1 = np.expand_dims(mask1,0)
        
        if action_dim > 2:
            mask2 = np.ones((output_dim,action_dim))
            mask2[:,2] = 0
            mask2 = np.expand_dims(mask2,0)
            masks = np.concatenate((mask0,mask1,mask2),0)
        else:
            masks = np.concatenate((mask0,mask1),0)
        #masks = np.concatenate((mask0,mask1),0)
        self.masks = tf.convert_to_tensor(masks[:,:,:self.action_dim],dtype=tf.float32)

        #self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        #self.train_loss = tf.metrics.mean(name='train_loss')
        #self.train_loss = tf.mean(name='train_loss')
        self.train_loss = tf.contrib.eager.metrics.Mean(name='train_loss')
        self.val_loss = tf.contrib.eager.metrics.Mean(name='val_loss')
    

    def _proj_u_onto_v(self, u, v):
        u_dot_v = tf.matmul(tf.expand_dims(u,1),tf.expand_dims(v,-1))
        v_dot_v = tf.matmul(tf.expand_dims(v,1),tf.expand_dims(v,-1))
        proj = u_dot_v/v_dot_v*tf.expand_dims(v,-1)
        return proj

    def _gram_schmidt(self, H):
        n_cols = H.shape[-1]
        for c in range(n_cols):
        #for c in range(1):
            # make col orthogonal to previous columns
            col = H[:,:,c]
            for prev_c in range(c):
                projection = tf.squeeze(self._proj_u_onto_v(col,H[:,:,prev_c]))  # project col onto previous column
                col = col - projection

            # scale column s.t. it has norm 1
            norm_of_col = tf.expand_dims(tf.norm(col,axis=-1),-1)
            e = col/norm_of_col
            H = col_replacement(H,c,e,self.masks)
        return H

    #@tf.function
    def _get_map(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.tanh(x)
        x = self.linear_2(x)
        x = tf.nn.tanh(x)
        x = self.linear_3(x)
        #print(x.shape)
        SCL_map = tf.reshape(x, [-1, self.output_dim, self.action_dim])
        #print(SCL_map.shape)
        #SCL_map = tf.reshape(x, [x.shape[0], self.output_dim, self.action_dim])
        SCL_map = self._gram_schmidt(SCL_map)
        #return x
        return SCL_map
    
    def _get_best_action(self, SCL_map, low_level_action):
        SCL_map_pinv = tf.transpose(SCL_map,perm=[0,2,1]) # assumes SCL_map is orthonormal
        low_level_action = tf.expand_dims(low_level_action,-1)
        #print(SCL_map_pinv)
        #print(low_level_action)
        least_square_sol = tf.matmul(SCL_map_pinv,low_level_action)
        return least_square_sol

    def get_config(self):
        return {"linear1":self.linear_1, "linear2": self.linear_2, "linear3": self.linear_3}
    
    def save_model_weights(self, weight_savepath):
        weight_dict = {'linear_1': [weight.tolist() for weight in self.linear_1.get_weights()], 
               'linear_2': [weight.tolist() for weight in self.linear_2.get_weights()],
               'linear_3': [weight.tolist() for weight in self.linear_3.get_weights()]}

        with open(weight_savepath+'.json','w+') as outfile:
                json.dump(weight_dict, outfile)
    
    def load_model_weights(self, weight_savepath):
        with open(weight_savepath + '.json', 'r') as f:
            loaded_weights = json.load(f)

        
        linear1_weights = [np.array(weight) for weight in loaded_weights['linear_1']]
        linear2_weights = [np.array(weight) for weight in loaded_weights['linear_2']]
        linear3_weights = [np.array(weight) for weight in loaded_weights['linear_3']]
        # give dummy input so linear layers can be built with appropriate number of parameters
        # cond_inp = tf.zeros((1,linear1_weights[0].shape[0]), dtype=tf.dtypes.float64)
        # qdot = tf.zeros((1,self.output_dim), dtype=tf.dtypes.float64)
        # print(cond_inp.shape)
        # print(qdot.shape)
        # qdot_hat = self.call(cond_inp, qdot)  # calling builds linear layers

        self.linear_1.set_weights(linear1_weights)
        self.linear_2.set_weights(linear2_weights)
        self.linear_3.set_weights(linear3_weights)

    def weight_proj(self, W, lam):
        scale_factor = 1/tf.stop_gradient(tf.math.maximum(1,tf.norm(W,ord=2)/lam)) # from Gouk et al Reg of NN by Enforcing Lipschitz Continuity
        # if scale_factor < 1:
        #     print('reprojected')
        
        return scale_factor*W

    #@tf.function
    def train_step(self, cond_inp, qdot):
        with tf.GradientTape() as tape:
            predictions = self.call(cond_inp, qdot)
            qdot = tf.expand_dims(qdot,-1)
            #loss = self.mse(qdot, predictions)
            loss = tf.losses.mean_squared_error(qdot, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        #print(g.trainable_variables)
        #self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # reproject weights to have lipschitz constant 1
        # this really should be a matter of the magnitudes of the inputs
        #old_w2 = copy.deepcopy(self.linear_2.w)
        #old_w1 = copy.deepcopy(self.linear_1.w)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        w1 = self.linear_1.get_weights()

        #print('\nlip coeff', self.lip_coeff)
        self.linear_1.set_weights([self.weight_proj(w1[0],self.lip_coeff),w1[1]])
        w2 = self.linear_2.get_weights()
        self.linear_2.set_weights([self.weight_proj(w2[0],self.lip_coeff),w2[1]])
        w3 = self.linear_3.get_weights()
        self.linear_3.set_weights([self.weight_proj(w3[0],self.lip_coeff),w3[1]])
        # self.linear_1.w = self.weight_proj(self.linear_1.w,1)
        # self.linear_2.w = self.weight_proj(self.linear_2.w,1)
        # self.linear_3.w = self.weight_proj(self.linear_3.w,1)
        #new_w2 = copy.deepcopy(self.linear_2.w)
        #new_w1 = copy.deepcopy(self.linear_1.w)
        # if tf.norm(new_w2-old_w2) > 0:
        #     print('w2 changed')
        # if tf.norm(new_w1-old_w1) > 0:
        #     print('w1 changed')
        
        self.train_loss(loss)
        #self.train_loss(loss)


    #@tf.function
    def val_step(self, cond_inp, qdot):
        predictions = self.call(cond_inp, qdot)
        qdot = tf.expand_dims(qdot,-1)
        #loss = self.mse(qdot, predictions)
        loss = tf.losses.mean_squared_error(qdot, predictions)

        #self.val_loss(loss)
        self.val_loss(loss)


    def call(self, cond_inp, low_level_action):
        SCL_map = self._get_map(cond_inp)
        
        # troubleshooting print statements
        # print(tf.norm(SCL_map,axis=1))
        # print(SCL_map[0,:,0].shape)
        # print(tf.tensordot(SCL_map[0,:,0],SCL_map[0,:,1],axes=1))
        # print(tf.tensordot(SCL_map[0,:,0],SCL_map[0,:,2],axes=1))
        # print(tf.tensordot(SCL_map[0,:,1],SCL_map[0,:,2],axes=1))


        a_star = self._get_best_action(SCL_map, low_level_action)
        lla_projection = tf.matmul(SCL_map, a_star)
        return lla_projection



