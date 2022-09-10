import json
import argparse
from pathlib import Path
import numpy as np
import random

def main(args):

    proportion_train = 0.8
    home = str(Path.home())
    datapath = home+args.demo_save_path
    with open(datapath + '.json', 'r') as f:
        data = json.load(f)
    
    total_examples = 0
    for traj in data:
        total_examples += len(traj)

    num_train_examples = round(proportion_train*total_examples)
    num_val_examples = total_examples - num_train_examples

    atomized_data = []
    for traj in data:
        for datapt in traj:
            atomized_data.append(datapt)

    assert len(atomized_data) == total_examples

    if args.trig_encode:
        # demo_data_10000_trig_encode-train
        train_savepath = datapath+'_trig_encode'+'-train'
        val_savepath = datapath+'_trig_encode'+'-val'

        encoded_data = []
        for datapt in atomized_data:
            thetas = np.array(datapt['thetas'])
            new_datapt = datapt
            new_datapt["thetas"] = list(np.concatenate((np.cos(thetas),np.sin(thetas))))
            
            encoded_data.append(new_datapt)

        assert len(encoded_data) == total_examples

        print('before shuffle:',encoded_data[0])
        random.shuffle(encoded_data)
        print('after shuffle:',encoded_data[0])

        train_data = encoded_data[:num_train_examples]
        val_data = encoded_data[num_train_examples:]
        assert len(val_data) == num_val_examples

        assert len(val_data) + len(train_data) == len(atomized_data)



        with open(train_savepath+'.json','w+') as outfile:
            json.dump(train_data, outfile)

        with open(val_savepath+'.json','w+') as outfile:
            json.dump(val_data, outfile)


    else:
        
        train_data = atomized_data[:num_train_examples]
        val_data = atomized_data[num_train_examples:]

        print(num_train_examples)
        print(len(train_data))
        print(len(val_data))
        print(len(atomized_data))
        assert len(val_data) == num_val_examples
        assert len(val_data) + len(train_data) == len(atomized_data)

        train_savepath = datapath+'-train'
        val_savepath = datapath+'-val'
        with open(train_savepath+'.json','w+') as outfile:
            json.dump(train_data, outfile)

        with open(val_savepath+'.json','w+') as outfile:
            json.dump(val_data, outfile)
    
    return train_savepath


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--demo_save_path', help='Path to save demonstrations to', default='/uAlberta/projects/SCL_wam/data/wam_sim/raw/demo_data_10000', type=str)
    parser.add_argument("--trig_encode", action='store_true', default=False,
            help="whether to use cosine and sine of angles")
    
    args = parser.parse_args()
    train_savepath = main(args)
