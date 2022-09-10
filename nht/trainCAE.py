from cgi import test
import tensorflow as tf
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("TensorFlow version:", tf.__version__)
tf.enable_eager_execution()

from CAE import CAE
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)



from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from utils import col_replacement, get_dataset

import json
import tensorflow as tf
from utils import common_arg_parser
from pathlib import Path

from tensorflow.python.ops import summary_ops_v2
import datetime

def get_cond_inp(sample, out_dim):
    
    if args.goal_cond:
        cond_inp = tf.cast(tf.concat((sample['target'],sample['x'],sample['q']),1),dtype=tf.float32)
    else:
        q = sample['q']
        
        q = q[:,:out_dim]
        
        noise = tf.random.normal((1,10),mean=0.0,stddev=0.05,dtype=tf.float32)
        
        cond_inp = tf.cast(tf.concat((sample['x'],q),1),dtype=tf.float32)
        cond_inp = cond_inp+noise

    return cond_inp

def get_dsets(args, batch_size):
    home = str(Path.home())
    datapath = home+args.demo_save_path
    train_datapath = datapath + '-train.tfrecord'
    train_ds = get_dataset(train_datapath)
    train_ds = train_ds.shuffle(10000).batch(batch_size)

    val_datapath = datapath + '-val.tfrecord'
    val_ds = get_dataset(val_datapath)
    val_ds = val_ds.shuffle(10000).batch(batch_size)
    
    return train_ds, val_ds

def config_loggers(args):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/CAE/testlog/' + current_time + '/train'
    test_log_dir = 'logs/CAE/testlog/' + current_time + '/val'
    train_summary_writer = summary_ops_v2.create_file_writer(train_log_dir)
    test_summary_writer = summary_ops_v2.create_file_writer(test_log_dir)

    return train_summary_writer, test_summary_writer

def main(args):

    out_dim = 7
    g = CAE(input_dim=out_dim, latent_dim=2, cond_dim=10, lip_coeff=None, hiddens=[64,64])
    #print(g.trainable_variables)

    # get datasets
    train_ds, val_ds = get_dsets(args, 256)

    # set up loggers
    train_summary_writer, test_summary_writer = config_loggers(args)


    EPOCHS = args.epochs
    for epoch in range(EPOCHS):
        
        # train
        for sample in train_ds:
            
            cond_inp = get_cond_inp(sample, out_dim)
            qdot = sample['q_dot']
            qdot = tf.cast(qdot,dtype=tf.float32)

            g.train_step(cond_inp, qdot)
            
        # log training progress
        with train_summary_writer.as_default(), summary_ops_v2.always_record_summaries():
            summary_ops_v2.scalar('loss', g.train_loss.result(), step=epoch)
            
        # evaluate
        for sample in val_ds:
            cond_inp = get_cond_inp(sample, out_dim)
            qdot = sample['q_dot']
            qdot = tf.cast(qdot,dtype=tf.float32)

            g.val_step(cond_inp, qdot)

        # log evaluation
        with test_summary_writer.as_default(), summary_ops_v2.always_record_summaries():
            summary_ops_v2.scalar('loss', g.val_loss.result(), step=epoch)

        # print losses
        print(
        f'Epoch {epoch + 1}, '
        f'Loss: {g.train_loss.result()}, '
        f'Val Loss: {g.val_loss.result()}, '
        )

    g.decoder.net.save('./testmodel_CAE')


    train_ds, val_ds = get_dsets(args, 1)
    for sample in val_ds.take(1):
        
        q = sample['q']
        cond_inp = tf.cast(tf.concat((sample['x'],q),1),dtype=tf.float32)
        qdot = sample['q_dot']
        qdot = tf.cast(qdot,dtype=tf.float32)

        z = g.encoder(tf.concat((qdot, cond_inp),axis=-1))
        qdot_hat = g.decoder(tf.concat((z, cond_inp),axis=-1))
        
        print('\nqdot\n', tf.expand_dims(qdot,-1))
        print('\nqdot_hat\n', tf.expand_dims(qdot_hat,-1))
        print('\nerror\n', tf.losses.mean_squared_error(qdot, qdot_hat))



    loaded_decoder = keras.models.load_model('./testmodel_CAE')
    qdot_hat = loaded_decoder(tf.concat((z, cond_inp),axis=-1))

    print('\n\n\n LOADED \n\n')
    print('\nqdot\n', tf.expand_dims(qdot,-1))
    print('\nqdot_hat\n', tf.expand_dims(qdot_hat,-1))
    print('\nerror\n', tf.losses.mean_squared_error(qdot, qdot_hat))


if __name__ == '__main__':
    arg_parser = common_arg_parser()
    args = arg_parser.parse_args()
    main(args)