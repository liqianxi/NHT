from cgi import test
import tensorflow as tf
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("TensorFlow version:", tf.__version__)
tf.enable_eager_execution()

from scl.CVAE import CVAE
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)



from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from scl.utils import get_cond_inp, get_dsets, config_loggers, get_model_dir, get_sci_str, get_reg_str

import json
import tensorflow as tf
from scl.utils import common_arg_parser
from pathlib import Path

from tensorflow.python.ops import summary_ops_v2
import datetime


def main(args):
    # map_range_dicts = {'alpha': map_step_size_range, 'beta': beta_range, 'units': units_per_hidden_range,
    #                 'batch_size': map_batch_range, 'hidden_layers': num_hidden_range, 'activation': activation_range}
    tf.compat.v1.set_random_seed(args.seed)
    latent_dim = 2
    out_dim = 7
    if args.override_params is not None:
        with open(args.override_params,'r') as f:
            override_params = json.load(f)

        print('read param json file')
        alpha = override_params['alpha']
        beta = override_params['beta']
        hiddens = [override_params['units']]*override_params['hidden_layers']
        activation = override_params['activation']
        batch_size = override_params['batch_size']
        g = CVAE(input_dim=out_dim, latent_dim=latent_dim, cond_dim=10, step_size = alpha, beta = beta, hiddens=hiddens, activation=activation)

    else:
        batch_size = 256
        g = CVAE(input_dim=out_dim, latent_dim=latent_dim, cond_dim=10, step_size = args.alpha, beta = args.reg)

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
        with train_summary_writer.as_default(), summary_ops_v2.always_record_summaries():
            summary_ops_v2.scalar('loss', g.train_loss.result(), step=epoch)
            summary_ops_v2.scalar('recon_loss', g.train_recon_loss.result(), step=epoch)
            summary_ops_v2.scalar('kl_loss', g.train_kl_loss.result(), step=epoch)
            
            
        # evaluate
        for sample in val_ds:
            cond_inp = get_cond_inp(sample, out_dim, args.goal_cond)
            qdot = sample['q_dot']
            qdot = tf.cast(qdot,dtype=tf.float32)

            g.val_step(cond_inp, qdot)

        # log evaluation
        with test_summary_writer.as_default(), summary_ops_v2.always_record_summaries():
            summary_ops_v2.scalar('loss', g.val_loss.result(), step=epoch)
            summary_ops_v2.scalar('recon_loss', g.val_recon_loss.result(), step=epoch)
            summary_ops_v2.scalar('kl_loss', g.val_kl_loss.result(), step=epoch)
        #print(g.steps)
        # print losses
        print(
        f'Epoch {epoch + 1}, '
        f'Loss: {g.train_loss.result()}, '
        f'Recon Loss: {g.train_recon_loss.result()}, '
        f'KL Loss: {g.train_kl_loss.result()}, '
        f'Val Loss: {g.val_loss.result()}, '
        )

    model_dir = get_model_dir(args)
    
    model_name = 'CVAE'
    g.decoder.net.save(f'{model_dir}/{model_name}')


    test_loading = True
    if test_loading:
        train_ds, val_ds = get_dsets(args, 1)
        for sample in val_ds.take(1):
            
            q = sample['q']
            cond_inp = tf.cast(tf.concat((sample['x'],q),1),dtype=tf.float32)
            qdot = sample['q_dot']
            qdot = tf.cast(qdot,dtype=tf.float32)

            z_params = g.encoder(tf.concat((qdot, cond_inp),axis=-1))
            mu = z_params[:,:latent_dim]
            sigma = z_params[:,latent_dim:]
            z = g.reparameterization(mu, sigma)
            qdot_hat = g.decoder(tf.concat((z, cond_inp),axis=-1))
            
            print('\nqdot\n', tf.expand_dims(qdot,-1))
            print('\nqdot_hat\n', tf.expand_dims(qdot_hat,-1))
            print('\nerror\n', tf.losses.mean_squared_error(qdot, qdot_hat))



        loaded_decoder = keras.models.load_model(f'{model_dir}/{model_name}')
        qdot_hat = loaded_decoder(tf.concat((z, cond_inp),axis=-1))

        print('\n\n\n LOADED \n\n')
        print('\nqdot\n', tf.expand_dims(qdot,-1))
        print('\nqdot_hat\n', tf.expand_dims(qdot_hat,-1))
        print('\nerror\n', tf.losses.mean_squared_error(qdot, qdot_hat))


if __name__ == '__main__':
    
    #print(tf.test.is_gpu_available())
    #input('wait')
    arg_parser = common_arg_parser()
    args = arg_parser.parse_args()
    main(args)