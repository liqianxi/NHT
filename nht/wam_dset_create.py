import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.enable_eager_execution()
from nht.utils import common_arg_parser
from pathlib import Path
import numpy as np
import json



def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    #array = tf.io.serialize_tensor(array)
    array = tf.serialize_tensor(array)
    return array

def parse_single_traj_pt(traj_pt, args):
    #define the dictionary -- the structure -- of our single example
    if args.zero_wrist:
        q_dot = traj_pt['velocity']
        q_dot[-3:] = np.zeros(3)
        traj_pt['velocity'] = q_dot

    data = {
        'x' : _bytes_feature(serialize_array(np.array(traj_pt['position']))),
        'target' : _bytes_feature(serialize_array(np.array(traj_pt['target']))),
        'q' : _bytes_feature(serialize_array(np.array(traj_pt['thetas']))),
        'q_dot' : _bytes_feature(serialize_array(np.array(traj_pt['velocity'])))
    }
    
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def parse_transition(traj_pt1, traj_pt2, args):
    #define the dictionary -- the structure -- of our single example
    
    data = {
        'x' : _bytes_feature(serialize_array(np.array(traj_pt1['position']))),
        'target' : _bytes_feature(serialize_array(np.array(traj_pt1['target']))),
        'q' : _bytes_feature(serialize_array(np.array(traj_pt1['thetas']))),
        'q_dot' : _bytes_feature(serialize_array(np.array(traj_pt1['velocity']))),
        'x_p' : _bytes_feature(serialize_array(np.array(traj_pt2['position']))),
        'q_p' : _bytes_feature(serialize_array(np.array(traj_pt2['thetas']))),
    }
    
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def write_data_to_tfr(data, args, filename:str="data"):
    filename= filename+".tfrecord"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0

    for traj_pt in data:
        #get the data we want to write
        out = parse_single_traj_pt(traj_pt, args)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

def write_transition_data_to_tfr(data, args, filename:str="data"):
    filename= filename+".tfrecord"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0

    Delta = args.Delta

    n = len(data)
    for i, traj_pt in enumerate(data):
        if i+1 >= n: # no next element available
            break
        # Inflate size of transitions
        # Small time-scale of simulation causes noise on inputs to wash out any detectable influence of the action on the next state
        q = np.array(traj_pt['thetas']).copy()
        q_p = np.array(data[i+1]['thetas']).copy()
        x = np.array(traj_pt['position']).copy()
        x_p = np.array(data[i+1]['position']).copy()
        
        inflated_q_p = q + Delta*(q_p-q)
        inflated_x_p = x + Delta*(x_p-x)

        inflated_next_traj_pt = data[i+1].copy()
        inflated_next_traj_pt['thetas'] = list(inflated_q_p)
        inflated_next_traj_pt['position'] = list(inflated_x_p)
        #print(traj_pt)
        #print(data[i+1])
        # print('vel', np.array(traj_pt['velocity']))
        # print('diff', np.array(inflated_next_traj_pt['thetas']) - np.array(traj_pt['thetas']))
        # print('vel-diff',np.array(traj_pt['velocity']) - (np.array(inflated_next_traj_pt['thetas']) - np.array(traj_pt['thetas'])) )
        # input('wait')
        #get the data we want to write
        out = parse_transition(traj_pt, inflated_next_traj_pt, args)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count



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


def get_dataset(filename, transition=False):
    #create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    #pass every single feature through our mapping function
    if transition:
        dataset = dataset.map(parse_transition_dset_element)
    else:
        dataset = dataset.map(parse_tfr_element)

    return dataset


def main(args):
    home = str(Path.home())
    datapath = home+args.demo_save_path
    
    train_datapath = datapath + '-train.json'
    with open(train_datapath, 'r') as f:
        train_data = json.load(f)

    val_datapath = datapath + '-val.json'
    with open(val_datapath, 'r') as f:
        val_data = json.load(f)

    if args.zero_wrist:
        datapath = datapath + '-zero_wrist'

    if args.transition_dset:
        write_transition_data_to_tfr(train_data, args, filename=datapath + "_transition" + '-train')
        write_transition_data_to_tfr(val_data, args, filename=datapath + "_transition" + '-val')
    
        train_dataset = get_dataset(datapath + "_transition" + '-train.tfrecord', transition=True)
        val_dataset = get_dataset(datapath + "_transition" + '-val.tfrecord', transition=True)

    else:
        write_data_to_tfr(train_data, args, filename=datapath + '-train')
        write_data_to_tfr(val_data, args, filename=datapath + '-val')

        train_dataset = get_dataset(datapath + '-train.tfrecord')
        val_dataset = get_dataset(datapath + '-val.tfrecord')

    for sample in train_dataset.take(2):
        print(sample)

    

    for sample in val_dataset.take(2):
        print(sample)


if __name__ == '__main__':
    arg_parser = common_arg_parser()
    args = arg_parser.parse_args()
    main(args)