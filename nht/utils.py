# disable deprecation and future warnings from tf1
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


import tensorflow as tf

#tf.compat.v1.enable_eager_execution
#tf.enable_eager_execution

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras
import numpy as np

from tensorflow.python.ops import summary_ops_v2
import datetime
from pathlib import Path
import os

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
  parser = arg_parser()
  parser.add_argument('--demo_save_path', type=str)
  parser.add_argument('--weight_savepath', type=str)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--action_dim', type=int, default=2)
  parser.add_argument('--goal_cond', action='store_true', default=False)
  parser.add_argument('--env_mode', type=str, default='normal')
  parser.add_argument('--reg', type=float, default=1e-3)
  parser.add_argument('--run', type=int, default=0)
  parser.add_argument('--sanity_check', action='store_true', default=False)
  parser.add_argument('--zero_wrist', action='store_true', default=False)
  parser.add_argument("--trig_encode", action='store_true', default=False,
            help="whether to use cosine and sine of angles")
  parser.add_argument('--transition_dset', action='store_true', default=False)
  parser.add_argument('--Delta', type=float, default=50.0)
  parser.add_argument('--model', type=str, default='LASER')
  parser.add_argument('--action_pred', action='store_true', default=False)
  parser.add_argument('--center', action='store_true', default=False)
  parser.add_argument('--check_norms', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--override_params', default=None, type=str)

  parser.add_argument('--alpha', type=float, default=1e-3)
  parser.add_argument('--lip_coeff', type=float, default=1.0)
  parser.add_argument('--units', type=int, default=256)
  parser.add_argument('--batch_size', type=int, default=256)
  parser.add_argument('--hidden_layers', type=int, default=2)
  parser.add_argument('--activation', type=str, default='tanh')

  parser.add_argument('--run_name', type=str, default='default')
  parser.add_argument('--legacy', action='store_true', default=False)

  parser.add_argument('--sweep', action='store_true', default=False)
  
  parser.add_argument('--u_dim', type=int, default=7)
  parser.add_argument('--o_dim', type=int, default=10)

  return parser



def kl_divergence(mu, sigma, target_sigma=1.0):
    var = tf.math.pow(sigma, 2)
    log_var = tf.math.log(var)
    KLD = -0.5*tf.math.reduce_sum(1 + log_var - tf.math.pow(mu,2) - var, axis=1) # this assumes identity covariance of target dist
    # target_var = tf.math.pow(target_sigma, 2)
    # log_target_var = tf.math.log(target_var)
    # KLD = 0.5*tf.math.reduce_sum(log_target_var - log_var, axis=1) + 1/(2*target_var)*tf.math.reduce_sum(var - target_var + tf.math.pow(mu,2), axis=1)

    return tf.math.reduce_mean(KLD)


def get_cond_inp(sample, out_dim, goal_cond, legacy=False):
  if legacy == True:
    return legacy_get_cond_inp(sample, out_dim, goal_cond)
  else:
    return new_get_cond_inp(sample, goal_cond)

# Legacy implementation for NHT CoRL initial submission
def legacy_get_cond_inp(sample, out_dim, goal_cond):
    
    if goal_cond:
        cond_inp = tf.cast(tf.concat((sample['target'],sample['x'],sample['q']),1),dtype=tf.float32)
    else:
        q = sample['q']
        
        q = q[:,:out_dim]
        
        noise = tf.random.normal((1,10),mean=0.0,stddev=0.05,dtype=tf.float32)
        #noise = tf.random.normal((1,10),mean=0.0,stddev=0.01,dtype=tf.float32)
        
        cond_inp = tf.cast(tf.concat((sample['x'],q),1),dtype=tf.float32)
        cond_inp = cond_inp+noise

    return cond_inp

def new_get_cond_inp(sample, goal_cond):
    
    if goal_cond:
        cond_inp = tf.cast(tf.concat((sample['target'],sample['x'],sample['q']),1),dtype=tf.float32)
    else:
        q = sample['q']
        
        #noise = tf.random.normal(shape=q.shape,mean=0.0,stddev=0.05,dtype=tf.float32)
        noise = tf.zeros_like(q, dtype=tf.float32)
        cond_inp = tf.cast(q, dtype=tf.float32)
        cond_inp = cond_inp+noise

    return cond_inp

# def get_cond_inp(sample, out_dim, goal_cond):
    
#     if goal_cond:
#         cond_inp = tf.cast(tf.concat((sample['target'],sample['x'],sample['q']),1),dtype=tf.float32)
#     else:
#         q = sample['q']
        
#         q = q[:,:out_dim]
        
#         noise = tf.random.normal((1,10),mean=0.0,stddev=0.05,dtype=tf.float32)
#         #noise = tf.random.normal((1,10),mean=0.0,stddev=0.01,dtype=tf.float32)
        
#         cond_inp = tf.cast(tf.concat((sample['x'],q),1),dtype=tf.float32)
#         cond_inp = cond_inp+noise

#     return cond_inp

def weight_proj(W, lam):
  scale_factor = 1/tf.stop_gradient(tf.math.maximum(1,tf.norm(W,ord=2)/lam)) # from Gouk et al Reg of NN by Enforcing Lipschitz Continuity
  # if scale_factor < 1:
  #     print('reprojected')
  
  return scale_factor*W

def project_weights(network, L):
    W_0 = network.get_weights().copy()
    for i, w in enumerate(W_0):
        if len(w.shape) > 1: # not bias weights
            projected_w = weight_proj(w, L)
            W_0[i] = projected_w
  
    network.set_weights(W_0)

def get_dsets(args, batch_size):
    home = str(Path.home())
    datapath = home+args.demo_save_path

    train_ds = get_dataset(datapath, suffix='-train.tfrecord', transition=args.transition_dset)
    train_ds = train_ds.shuffle(10000).batch(batch_size)
    
    val_ds = get_dataset(datapath, suffix='-val.tfrecord', transition=args.transition_dset)
    val_ds = val_ds.shuffle(10000).batch(batch_size)
    
    return train_ds, val_ds

def get_dataset(datapath, suffix, transition=False):

  if transition:
    datapath = datapath + "_transition"
  filename = datapath + suffix

  #create the dataset
  dataset = tf.data.TFRecordDataset(filename)

  #pass every single feature through our mapping function
  if transition:
      dataset = dataset.map(parse_transition_dset_element)
  else:
      dataset = dataset.map(parse_tfr_element)

  return dataset

def get_model_dir(args):
  #model_dir = f'./models/run_{args.run}'
  model_dir = 'map_model'
  if not os.path.exists(model_dir):
      os.makedirs(model_dir)
  return model_dir

def get_local_model_dir(args):
  #model_dir = f'./models/run_{args.run}'
  # --weight_savepath=grasp_o --model=SVD_uncentered
  model_dir = f'models/{args.weight_savepath}/{args.model}'
  if not os.path.exists(model_dir):
      os.makedirs(model_dir)
  return model_dir

def get_sci_str(value):
  # returns scientific notation of input in string form (assuming smaller than 1)
  decimal = str(value).split('.')[1]
  return f'{decimal[-1]}e-{len(decimal)}'

def get_decimal_str(value):
  integer = int(str(value).split('.')[0])
  decimal = int(str(value).split('.')[1])

  return f'{integer}p{decimal}'

def get_reg_str(value):
  if value > 1:
    reg_str = get_decimal_str(value)
  else:
    reg_str = get_sci_str(value)
  
  return reg_str

def config_loggers(args):
  
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  #train_log_dir = f'logs/run_{args.run}/alpha_{get_sci_str(args.alpha)}-reg_{args.reg}/' + current_time + '/train'
  #test_log_dir = f'logs/run_{args.run}/alpha_{get_sci_str(args.alpha)}-reg_{args.reg}/' + current_time + '/val'
  # train_log_dir = f'logs/run_{args.run}/alpha_{args.alpha}-reg_{args.reg}/' + current_time + '/train'
  # test_log_dir = f'logs/run_{args.run}/alpha_{args.alpha}-reg_{args.reg}/' + current_time + '/val'
  train_log_dir = 'map_logs/' + current_time + '/train'
  test_log_dir = 'map_logs/' + current_time + '/val'
  train_summary_writer = summary_ops_v2.create_file_writer(train_log_dir)
  test_summary_writer = summary_ops_v2.create_file_writer(test_log_dir)

  print('\nLogging to '+ train_log_dir+'\n')
  return train_summary_writer, test_summary_writer


def col_replacement(original_tensor, col_idx, newcol, masks):
  # Create mask in numpy then convert to tensor
  #print(original_tensor.shape)
  #print(masks.shape)
  #print(masks[0,:,:])
  #print('\n\n\ncol idx', col_idx)
  mask = masks[col_idx,:,:]  # has zeros in col corresponding to col_idx
  dims = original_tensor.shape
  #print(dims)
  right_vec = np.zeros((1,1,dims[-1]))  # batch of (1 x cols) row vectors, where cols is number of cols in original_tensor
  right_vec[:,0,col_idx] = 1.0  # set value at index of column we want to replace to 1.0
  right_vec_tensor = tf.convert_to_tensor(right_vec, dtype=tf.float32)
  #right_vec_tensor = right_vec_tensor*tf.ones((tf.shape(original_tensor)[0],1,dims[-1]))
  
  test_vec = tf.expand_dims(tf.ones_like(original_tensor[:,0,:]),1)

  right_vec_tensor = right_vec_tensor*test_vec
  left_vec_tensor = tf.expand_dims(newcol,-1)
  newcol_tensor = tf.matmul(left_vec_tensor,right_vec_tensor)  # newcol_tensor = [new col]*[1 0 0 ... 0] where [new col] is a column vector
  #newcol_tensor = tf.concat((tf.expand_dims(newcol,-1),tf.expand_dims(newcol,-1)),-1)
  #print(newcol_tensor.shape)
  #print(newcol.shape)
  #print(newcol_mat[:,:,col_idx].shape)
  #print(newcol_mat[:,:,col_idx])
  #print(newcol_tensor)

  #newcol_mat[:,:,col_idx] = newcol
  #newcol_tensor = tf.convert_to_tensor(newcol_mat)
  
  #new_tensor = original_tensor*tensor_mask + newcol_tensor*(1-mask)
  new_tensor = original_tensor*mask + newcol_tensor*(1-mask)
  #print('end of col replace', new_tensor.shape)
  return new_tensor



  #out = original_tensor
  #print(out.shape)
  #print(out)
  return out
  #mask = np.ones(original_tensor.shape[1:])
  #mask = tf.ones(original_tensor.shape)
  #mask = tf.ones(original_tensor.shape)
  #mask = tf.Variable(tf.ones_like(original_tensor))
  #mask[:,:,col_idx].assign(0)
  #print(mask)

  # i0 = tf.constant(0)
  # m0 = tf.ones([2, 2])
  # c = lambda i, m: i < 10
  # b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
  # output = tf.while_loop(
  #     c, b, loop_vars=[i0, m0],
  #     shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])



  # indices = [col_idx]*original_tensor.shape[1]
  # one_hot = tf.one_hot(indices, depth=2)
  # mask = tf.expand_dims(1-one_hot,0)
  # i0 = tf.constant(1)
  # m0 = tf.ones([1, 2])
  # print(tf.shape(original_tensor)[0])
  # c = lambda i, m: i < tf.shape(original_tensor)[0]
  # b = lambda i, m: [i+1, tf.concat([m, mask], axis=0)]
  # output = tf.while_loop(
  #     c, b, loop_vars=[i0, mask],
  #     shape_invariants=[i0.get_shape(), tf.TensorShape([None, original_tensor.shape[1],original_tensor.shape[2]])])
  
  return output[1]
  #c = lambda i, m: i < 2
  #c = lambda i, m: i < 10
  
  #b = lambda i, m: [i+1, tf.concat([m, m0], axis=0)]
  
  # print(output[1].eval())
  # print(output[1].eval().shape)
  #print(mask)
  
  # print(indices)
  
  # print(one_hot.shape)
  # print(one_hot.eval())
  #print((1-one_hot).eval())
  #ident = tf.eye(2, batch_shape=[original_tensor.shape[0]])
  #test = one_hot*ident
  #print(test.shape)
  #print(test.eval())
  

  
  #print(tf.shape(original_tensor)[0].eval())
  
  mask[:,:,col_idx] = 0
  #mask[:,col_idx] = 0
  tensor_mask = tf.convert_to_tensor(mask, dtype=tf.float32)

  # # Need matrix with same shape as original tensor. Only the values in the col we are replacing matter. Other entries are multiplied by 0 from (1-mask)
  # #print(tensor_mask.shape)
  # #print(newcol.shape)
  # #newcol_mat = tf.expand_dims(newcol,-1)
  # # use outer product to compute matrix with 0s everywhere except the column we want to replace
  # dims = original_tensor.shape
  # #right_vec = np.zeros((dims[0],1,dims[-1]))  # batch of (1 x cols) row vectors, where cols is number of cols in original_tensor
  # right_vec = np.zeros((1,dims[-1]))
  # #right_vec[:,0,col_idx] = 1.0  # set value at index of column we want to replace to 1.0
  # right_vec[0,col_idx] = 1.0  # set value at index of column we want to replace to 1.0
  # right_vec_tensor = tf.convert_to_tensor(right_vec, dtype=tf.float32)
  # newcol_tensor = tf.matmul(tf.expand_dims(newcol,-1),right_vec_tensor)  # newcol_tensor = [new col]*[1 0 0 ... 0] where [new col] is a column vector
  # #newcol_tensor = tf.concat((tf.expand_dims(newcol,-1),tf.expand_dims(newcol,-1)),-1)
  # #print(newcol_tensor.shape)
  # #print(newcol.shape)
  # #print(newcol_mat[:,:,col_idx].shape)
  # #print(newcol_mat[:,:,col_idx])
  # #print(newcol_tensor)

  # #newcol_mat[:,:,col_idx] = newcol
  # #newcol_tensor = tf.convert_to_tensor(newcol_mat)
  
  # #new_tensor = original_tensor*tensor_mask + newcol_tensor*(1-mask)
  # new_tensor = original_tensor*tensor_mask + newcol_tensor*(1-mask)
  
  # return new_tensor


def parse_tfr_element(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'x' : tf.io.FixedLenFeature([], tf.string),
      'target' : tf.io.FixedLenFeature([], tf.string),
      'q' : tf.io.FixedLenFeature([], tf.string),
      'q_dot' : tf.io.FixedLenFeature([], tf.string),
    }

    
  content = tf.io.parse_single_example(element, data)
  
  x = content['x']
  target = content['target']
  q = content['q']
  q_dot = content['q_dot']
  
  
  #get our 'feature'-- our image -- and reshape it appropriately
  x_feature = tf.io.parse_tensor(x, out_type=tf.double)
  target_feature = tf.io.parse_tensor(target, out_type=tf.double)
  q_feature = tf.io.parse_tensor(q, out_type=tf.double)
  q_dot_feature = tf.io.parse_tensor(q_dot, out_type=tf.double)
  return {'x': x_feature, 'target':target_feature, 'q':q_feature, 'q_dot':q_dot_feature}

def parse_transition_dset_element(element):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'x' : tf.io.FixedLenFeature([], tf.string),
        'target' : tf.io.FixedLenFeature([], tf.string),
        'q' : tf.io.FixedLenFeature([], tf.string),
        'q_dot' : tf.io.FixedLenFeature([], tf.string),
        'x_p' : tf.io.FixedLenFeature([], tf.string),
        'q_p' : tf.io.FixedLenFeature([], tf.string),
    }


    content = tf.io.parse_single_example(element, data)

    x = content['x']
    target = content['target']
    q = content['q']
    q_dot = content['q_dot']
    x_p = content['x_p']
    q_p = content['q_p']


    #get our 'feature'-- our image -- and reshape it appropriately
    x_feature = tf.io.parse_tensor(x, out_type=tf.double)
    target_feature = tf.io.parse_tensor(target, out_type=tf.double)
    q_feature = tf.io.parse_tensor(q, out_type=tf.double)
    q_dot_feature = tf.io.parse_tensor(q_dot, out_type=tf.double)
    x_p_feature = tf.io.parse_tensor(x_p, out_type=tf.double)
    q_p_feature = tf.io.parse_tensor(q_p, out_type=tf.double)
    return {'x': x_feature, 'target':target_feature, 'q':q_feature, 'q_dot':q_dot_feature, 'x_p': x_p_feature, 'q_p':q_p_feature}



