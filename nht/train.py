from cgi import test
import tensorflow as tf
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("TensorFlow version:", tf.__version__)
tf.enable_eager_execution()


from SCL_householder_old import SCL
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

def get_cond_inp(sample, out_dim):
    #cond_inp = tf.cast(tf.concat((sample['target'],sample['x'],sample['q']),1),dtype=tf.float32)
    if args.goal_cond:
        cond_inp = tf.cast(tf.concat((sample['target'],sample['x'],sample['q']),1),dtype=tf.float32)
    else:
        q = sample['q']
        #print(q.shape)
        q = q[:,:out_dim]
        #print(q.shape)
        noise = tf.random.normal((1,10),mean=0.0,stddev=0.05,dtype=tf.float32)
        #print(noise.shape)
        #print(noise)
        cond_inp = tf.cast(tf.concat((sample['x'],q),1),dtype=tf.float32)
        cond_inp = cond_inp+noise

    return cond_inp


def main(args):
    print(args.action_dim)
    if args.env_mode == '4dof':
        out_dim = 4
    else:
        out_dim = 7

    if args.goal_cond:
        cond_size = 13
    else:
        cond_size = out_dim+3

    g = SCL(action_dim=args.action_dim, output_dim=out_dim, cond_dim=cond_size, lip_coeff=args.lip_coeff)
    #g.load_model_weights("/home/kerrick/uAlberta/projects/SCL_wam/test_weights")

    '''
    embed_dim = 256  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer


    def create_model():
        inputs_tokens = layers.Input(shape=(maxlen,), dtype=tf.int32)
        inputs_category = layers.Input(shape=(1,), dtype=tf.int32, name="inputs_category")
        # the first custom layer for embedding 
        embedding_layer = TokenPositionAndCategoricalEmbedding(maxlen, vocab_size, 
                                                            number_of_categories, 
                                                            embed_dim)
        x = embedding_layer(inputs_tokens,inputs_category)
        # the second custom layer for GPT-kind-of transformer decoder block
        transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
        x = transformer_block(x)
        outputs = layers.Dense(vocab_size)(x)
        model = keras.Model(inputs=[inputs_tokens,inputs_category], outputs=[outputs, x])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(
            "adam", loss=[loss_fn, None],
        )  # No loss and optimization based on word embeddings from transformer block
        return model
    my_model=create_model()


    '''
    home = str(Path.home())
    datapath = home+args.demo_save_path
    train_datapath = datapath + '-train.tfrecord'
    train_ds = get_dataset(train_datapath)
    train_ds = train_ds.shuffle(10000).batch(256)

    val_datapath = datapath + '-val.tfrecord'
    val_ds = get_dataset(val_datapath)
    val_ds = val_ds.shuffle(10000).batch(256)

    # for sample in val_ds.take(1):
    #     #print(sample)
    #     cond_inp_sample = tf.concat((sample['x'],sample['q']),1)
    #     print(cond_inp_sample.shape[-1])
    #     print(cond_inp_sample.dtype)
    #     qdot_sample = sample['q_dot']
    #     print(qdot_sample.shape[-1])

    # def create_model():
    #     cond_inp = layers.Input(shape=(14))
    #     qdot = layers.Input(shape=(7))
    #     # the first custom layer for embedding 
    #     output = SCL(action_dim=2, output_dim=7)(cond_inp, qdot)
    #     model = keras.Model(inputs=[cond_inp, qdot], outputs=output)
    #     return model

    # g=create_model()


    # mse = tf.keras.losses.MeanSquaredError()
    # optimizer = tf.keras.optimizers.Adam()

    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # val_loss = tf.keras.metrics.Mean(name='val_loss')

    # @tf.function
    # def train_step(cond_inp, qdot):
    #     with tf.GradientTape() as tape:
    #         predictions = g(cond_inp, qdot)
    #         qdot = tf.expand_dims(qdot,-1)
    #         loss = mse(qdot, predictions)

    #     gradients = tape.gradient(loss, g.trainable_variables)
    #     #print(g.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, g.trainable_variables))

    #     train_loss(loss)


    # @tf.function
    # def val_step(cond_inp, qdot):
    #     predictions = g(cond_inp, qdot)
    #     qdot = tf.expand_dims(qdot,-1)
    #     loss = mse(qdot, predictions)

    #     val_loss(loss)

    # set up loggers
    from tensorflow.python.ops import summary_ops_v2
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/SCL/testlog2/' + current_time + '/train'
    test_log_dir = 'logs/SCL/testlog2/' + current_time + '/val'
    train_summary_writer = summary_ops_v2.create_file_writer(train_log_dir)
    test_summary_writer = summary_ops_v2.create_file_writer(test_log_dir)


    EPOCHS = args.epochs


    for epoch in range(EPOCHS):
        
        # Reset the metrics at the start of the next epoch
        #g.train_loss.reset_states()
        #g.val_loss.reset_states()

        # train
        for sample in train_ds:
            #cond_inp = tf.concat((sample['target'],sample['x'],sample['q']),1)
            cond_inp = get_cond_inp(sample, out_dim)
            if args.env_mode == '4dof':
                qdot = sample['q_dot'][:, :out_dim]
            else:
                qdot = sample['q_dot']

            qdot = tf.cast(qdot,dtype=tf.float32)
            g.train_step(cond_inp, qdot)
            
            

        # log training progress
        with train_summary_writer.as_default(), summary_ops_v2.always_record_summaries():
            summary_ops_v2.scalar('loss', g.train_loss.result(), step=epoch)
            

        # evaluate
        for sample in val_ds:
            cond_inp = get_cond_inp(sample, out_dim)
            if args.env_mode == '4dof':
                qdot = sample['q_dot'][:, :out_dim]
            else:
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



    # following block apparently does not work because I use eager execution
    # with train_summary_writer.as_default(), summary_ops_v2.always_record_summaries():
    #     summary_ops_v2.trace_on(graph=True, profiler=True)
    #     # call function
    #     summary_ops_v2.trace_export(
    #         name="my_func_trace",
    #         step=0,
    #         profiler_outdir=train_log_dir)

    #keras.experimental.export_saved_model(g, "my_scl_model", serving_only=True)
    #g.save_weights('test_weights')
    # print example input/output pair


    # print('weights')
    # w1 = g.linear_1.get_weights()
    # print(len(w1))
    # print(w1[0].shape)
    # print('endweights)')

    # weight_dict = {'linear_1': [weight.tolist() for weight in g.linear_1.get_weights()], 
    #                'linear_2': [weight.tolist() for weight in g.linear_2.get_weights()],
    #                'linear_3': [weight.tolist() for weight in g.linear_3.get_weights()]}

    # import json
    weight_savepath = args.weight_savepath
    #weight_savepath = 'fixedgoal_noisystarts'
    w1 = g.linear_1.w
    w2 = g.linear_2.w
    w3 = g.linear_3.w
    g.save_model_weights(weight_savepath)
    # with open(weight_savepath+'.json','w+') as outfile:
    #             json.dump(weight_dict, outfile)

    val_ds = get_dataset(val_datapath)
    example_ds = val_ds.batch(1)
    for sample in val_ds.take(1):
        #print(sample)
        print(sample['x'])
        print(sample['q'])
        #cond_inp = tf.expand_dims(get_cond_inp(sample),0)
        if args.goal_cond:
            cond_inp = tf.expand_dims(tf.cast(tf.concat((sample['target'],sample['x'],sample['q']),0),dtype=tf.float32),0)
        else:
            if args.env_mode == '4dof':
                q = sample['q']
                #print(q.shape)
                q = q[:out_dim]
            else:
                q = sample['q']
            cond_inp = tf.expand_dims(tf.cast(tf.concat((sample['x'],q),0),dtype=tf.float32),0)
        
        if args.env_mode == '4dof':
            qdot = sample['q_dot'][:out_dim]
        else:
            qdot = sample['q_dot']
            
        qdot = tf.expand_dims(tf.cast(qdot,dtype=tf.float32),0)
        qdot_hat = g(cond_inp, qdot)
        SCL_map = g._get_map(cond_inp)
        a_star = g._get_best_action(SCL_map, qdot)
        print('\nqdot\n', tf.expand_dims(qdot,-1))
        print('\nSCL map\n', SCL_map)
        print('\na*\n', a_star)
        print('\nqdot_hat\n', qdot_hat)
        print('\nerror\n', tf.losses.mean_squared_error(tf.expand_dims(qdot,-1), qdot_hat))

        
        # with train_summary_writer.as_default(), summary_ops_v2.always_record_summaries(): 
        #     summary_ops_v2.trace_on(graph=True, profiler=True)
        #     train_step(cond_inp, qdot)
        #     summary_ops_v2.trace_export(
        #         name="my_func_trace",
        #         step=0,
        #         profiler_outdir=train_log_dir)


    # with open(weight_savepath + '.json', 'r') as f:
    #     loaded_weights = json.load(f)

    # loaded_SCL = SCL(action_dim=2, output_dim=7)
    # # pass one input to initialize weight sizes
    # qdot_hat = loaded_SCL(cond_inp, qdot) 

    # loaded_SCL.linear_1.set_weights([np.array(weight) for weight in loaded_weights['linear_1']])
    # loaded_SCL.linear_2.set_weights([np.array(weight) for weight in loaded_weights['linear_2']])
    # loaded_SCL.linear_3.set_weights([np.array(weight) for weight in loaded_weights['linear_3']])

    loaded_SCL = SCL(action_dim=args.action_dim, output_dim=out_dim, cond_dim=cond_size, lip_coeff=args.lip_coeff)
    loaded_SCL.load_model_weights(weight_savepath)
    loaded_w1 = loaded_SCL.linear_1.w
    loaded_w2 = loaded_SCL.linear_2.w
    loaded_w3 = loaded_SCL.linear_3.w
    # print('norms')
    # print(tf.norm(w1,ord=2))
    # print(tf.norm(w2,ord=2))
    # print(tf.norm(w3,ord=2))
    # print('weight diff')
    # print(tf.norm(loaded_w1-w1,ord=2))
    # print(tf.norm(loaded_w2-w2,ord=2))
    # print(tf.norm(loaded_w3-w3,ord=2))
    # print(loaded_w1-w1)
    # print(loaded_w2-w2)
    # print(loaded_w3-w3)
    #loaded_scl = tf.compat.v1.keras.experimental.load_from_saved_model("my_scl_model", custom_objects={"CustomModel": SCL})
    print('\n\n\n LOADED \n\n')
    qdot_hat = loaded_SCL(cond_inp, qdot)
    SCL_map = loaded_SCL._get_map(cond_inp)
    a_star = loaded_SCL._get_best_action(SCL_map, qdot)
    print('\nqdot\n', tf.expand_dims(qdot,-1))
    print('\nSCL map\n', SCL_map)
    print('\na*\n', a_star)
    print('\nqdot_hat\n', qdot_hat)
    print('\nerror\n', tf.losses.mean_squared_error(tf.expand_dims(qdot,-1), qdot_hat))


if __name__ == '__main__':
    arg_parser = common_arg_parser()
    args = arg_parser.parse_args()
    main(args)