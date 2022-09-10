from cgi import test
import tensorflow as tf
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("TensorFlow version:", tf.__version__)
tf.enable_eager_execution()

from expNHT import NHT
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



def write_training_progress_to_logs(train_summary_writer, g, epoch):
    with train_summary_writer.as_default(), summary_ops_v2.always_record_summaries():
            summary_ops_v2.scalar('recon_loss', g.train_loss.result(), step=epoch)

def write_val_progress_to_logs(test_summary_writer, g, epoch):
    with test_summary_writer.as_default(), summary_ops_v2.always_record_summaries():
            summary_ops_v2.scalar('recon_loss', g.val_loss.result(), step=epoch)

def main(args):

    tf.compat.v1.set_random_seed(args.seed)
    out_dim = 7
    cond_dim = 10

    # Set hyperparameters
    if args.override_params is not None:   # here hyperparameters are read from an override_params file
        with open(args.override_params,'r') as f:
            override_params = json.load(f)

        print('read param json file')
        alpha = override_params['alpha']
        L = override_params['L']
        hiddens = [override_params['units']]*override_params['hidden_layers']
        activation = override_params['activation']
        batch_size = override_params['batch_size']
        g = NHT(action_dim=args.action_dim, output_dim=out_dim, cond_dim=cond_dim, step_size = alpha, lip_coeff = L, hiddens=hiddens, activation=activation)

    else:                      # hyperparameters from command line args
        alpha = args.alpha
        L = args.lip_coeff
        units = args.units
        batch_size = args.batch_size
        hidden_layers = args.hidden_layers
        activation = args.activation
        hiddens = [units]*hidden_layers
        g = NHT(action_dim=args.action_dim, output_dim=out_dim, cond_dim=cond_dim, step_size = alpha, lip_coeff = L, hiddens=hiddens, activation=activation)

    
    # get datasets
    train_ds, val_ds = get_dsets(args, batch_size)

    # set up loggers
    train_summary_writer, test_summary_writer = config_loggers(args)


    EPOCHS = args.epochs
    for epoch in range(EPOCHS):
        
        # train
        for sample in train_ds:
            
            cond_inp = get_cond_inp(sample, out_dim, args.goal_cond)
            qdot = sample['q_dot']
            qdot = tf.cast(qdot,dtype=tf.float32)

            g.train_step(cond_inp, qdot)
            
        # log training progress
        write_training_progress_to_logs(train_summary_writer, g, epoch)
            
        # evaluate
        for sample in val_ds:
            cond_inp = get_cond_inp(sample, out_dim, args.goal_cond)
            qdot = sample['q_dot']
            qdot = tf.cast(qdot,dtype=tf.float32)

            g.val_step(cond_inp, qdot)

        # log evaluation
        write_val_progress_to_logs(test_summary_writer, g, epoch)

        print(
        f'Epoch {epoch + 1}, '
        f'Train Loss: {g.train_loss.result()}, '
        f'Val Loss: {g.val_loss.result()}, '
        )
    

    model_dir = get_model_dir(args)
    
    model_name = 'SCL'
    g.h.net.save(f'{model_dir}/{model_name}')

    # to test we can load the saved model and get consistent results
    test_loading = True
    if test_loading:
        train_ds, val_ds = get_dsets(args, 1)
        for sample in val_ds.take(1):
            
            q = sample['q']
            cond_inp = tf.cast(tf.concat((sample['x'],q),1),dtype=tf.float32)
            qdot = sample['q_dot']
            qdot = tf.cast(qdot,dtype=tf.float32)

            SCL_map = g._get_map(cond_inp)
            a_star = g._get_best_action(SCL_map, qdot)
            qdot_hat = tf.matmul(SCL_map, a_star)
            error = tf.expand_dims(qdot,-1)-qdot_hat

            print('\nqdot\n', tf.expand_dims(qdot,-1))
            print('\nSCL map\n', SCL_map)
            print('\na*\n', a_star)
            print('\nqdot_hat\n', qdot_hat)
            print('\nerror\n', error)
            print('\nerror norm\n', tf.norm(error, axis=1))


        loaded_SCL = NHT(action_dim=args.action_dim, output_dim=out_dim, cond_dim=cond_dim, lip_coeff=args.lip_coeff, action_pred=args.action_pred)
        loaded_h = keras.models.load_model(f'{model_dir}/{model_name}')
        loaded_SCL.h = loaded_h
        
        SCL_map = loaded_SCL._get_map(cond_inp)
        a_star = loaded_SCL._get_best_action(SCL_map, qdot)
        qdot_hat = tf.matmul(SCL_map, a_star)
        error = tf.expand_dims(qdot,-1)-qdot_hat

        print('\nqdot\n', tf.expand_dims(qdot,-1))
        print('\nSCL map\n', SCL_map)
        print('\na*\n', a_star)
        print('\nqdot_hat\n', qdot_hat)
        print('\nerror\n', error)
        print('\nerror norm\n', tf.norm(error, axis=1))


if __name__ == '__main__':
    
    #print(tf.test.is_gpu_available())
    #input('wait')
    arg_parser = common_arg_parser()
    args = arg_parser.parse_args()
    main(args)