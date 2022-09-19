'''
July 13 2022

This calibration relies on a particular trained NHT/SCL model. 
It aims to solve the optimization problem:

argmax_Q e_1^T*J*H*Q*e_1

where H comes from the trained NHT model evaluated at the start state for the task, 
and J is the Jacobian.

Q is a constant matrix meant to rotate the users input during teleoperation. 
This basically ensures that, from the start state, pushing the joystick forward 
will result in the actuation that most moves the end-effector forward. 
'''

from cgi import test
import tensorflow as tf
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("TensorFlow version:", tf.__version__)
tf.enable_eager_execution()

from nht.NHT import SCL
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from nht.utils import common_arg_parser
from tensorflow.python.ops import summary_ops_v2

from scl.utils import get_cond_inp, get_dsets, config_loggers, get_model_dir

from gym.envs.robotics.utils import get_J, WAM_fwd_kinematics


def write_training_progress_to_logs(train_summary_writer, g, epoch):
    with train_summary_writer.as_default(), summary_ops_v2.always_record_summaries():
            summary_ops_v2.scalar('recon_loss', g.train_loss.result(), step=epoch)

def write_val_progress_to_logs(test_summary_writer, g, epoch):
    with test_summary_writer.as_default(), summary_ops_v2.always_record_summaries():
            summary_ops_v2.scalar('recon_loss', g.val_loss.result(), step=epoch)

def main(args):

    tf.compat.v1.set_random_seed(args.seed)
    out_dim = 2
    cond_dim = 1
    if args.override_params is not None:
        with open(args.override_params,'r') as f:
            override_params = json.load(f)

        print('read param json file')
        alpha = override_params['alpha']
        L = override_params['L']
        hiddens = [override_params['units']]*override_params['hidden_layers']
        activation = override_params['activation']
        batch_size = override_params['batch_size']
        g = SCL(action_dim=args.action_dim, output_dim=out_dim, cond_dim=cond_dim, step_size = alpha, lip_coeff = L, hiddens=hiddens, activation=activation)

    else:
        alpha = 0.001
        L = None
        units = 32
        batch_size = 1
        hidden_layers = 1
        activation = 'tanh'
        hiddens = [units]*hidden_layers
        g = SCL(action_dim=args.action_dim, output_dim=out_dim, cond_dim=cond_dim, step_size = alpha, lip_coeff = L, hiddens=hiddens, activation=activation)


    # set up loggers
    train_summary_writer, test_summary_writer = config_loggers(args)

    def get_cond_inp(q):

        x = WAM_fwd_kinematics(q, 7).copy()
        cond_inp = np.expand_dims(np.concatenate((x,q),0),0)

        return cond_inp

    # Get J_0 and H_hat_0
    q0 = [-2.02362697e-01,  4.33858512e-01,  4.71274239e-02,  2.17864115e+00,
       -3.92220801e-02,  5.29857799e-01, -1.25754495e-01]
    [J_0, T_ito0] = get_J(q0, n_joints=7)

    J_0 = tf.cast(J_0,dtype=tf.float32)

    NHT_cond_size = 10
    NHT_path = '/home/kerrick/uAlberta/projects/NHT/map_model/SCL'
    NHT_model = SCL(action_dim=2, output_dim=7, cond_dim=NHT_cond_size)
    NHT_model.h = tf.keras.models.load_model(NHT_path)
    

    #self.SCL_map = self.SCL_model._get_map(tf.stop_gradient(self.cond_inp))
    cond_inp0 = tf.cast(get_cond_inp(q0),dtype=tf.float32)
    H_hat_0 = NHT_model._get_map(tf.stop_gradient(cond_inp0))
    H_hat_0 = tf.squeeze(H_hat_0)


    EPOCHS = args.epochs
    for epoch in range(EPOCHS):
        
        # train
        unit_inp = tf.ones((1,1),dtype=tf.float32)
        g.input_rot_train_step(unit_inp, J_0, H_hat_0)
            
        # log training progress
        write_training_progress_to_logs(train_summary_writer, g, epoch)
        
        print(
        f'Epoch {epoch + 1}, '
        f'Train Loss: {g.train_loss.result()}, '
        )
    



    model_dir = 'map_model'
    
    model_name = 'teleop_alignment.json'
    

    e_2_pose_T = tf.expand_dims(tf.eye(6)[:,1],0)
    
    e_1_action = tf.eye(out_dim,1)

    Q = tf.squeeze(g._get_map(unit_inp))
    rotated_action = tf.matmul(Q, e_1_action)
    q_vel = tf.matmul(H_hat_0, rotated_action)
    x_vel = tf.matmul(J_0, q_vel)
    x2_vel = tf.matmul(e_2_pose_T, x_vel)

    x2_vel = x2_vel.numpy()
    Q = Q.numpy()
    if x2_vel < 0.0:
        Q[:,1] = -1*Q[:,1]

    alignment_dict = {'Q':Q.tolist()}
    with open(f'{model_dir}/{model_name}','w+') as outfile:
        json.dump(alignment_dict, outfile)



if __name__ == '__main__':
    
    #print(tf.test.is_gpu_available())
    #input('wait')
    arg_parser = common_arg_parser()
    args = arg_parser.parse_args()
    main(args)