# Modified by Michael Przystupa and Kerrick Johnstonbaugh
# code is pulling snippets largely from here:
# source: https://raw.githubusercontent.com/pyro-ppl/pyro/dev/examples/vae/vae.py

import argparse

from nht.utils import get_model
from nht.models.callbacks import LipschitzRegularizer
from nht.utils import ActionInterfaceDataset, split_dset
from torch.utils.data import DataLoader
import time 
import pytorch_lightning as pl
from pytorch_lightning import Trainer 
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
import gym
import d4rl
import numpy as np

def main(args):

    pl.seed_everything(args.rnd_seed)

    if args.d4rl_dset is not None:
        env = gym.make(args.d4rl_dset)
        data_dict = env.get_dataset()
        train_data_dict, val_data_dict = split_dset(data_dict, context=args.context, val_prop=args.val_prop)

    #create datasets
    train_dataset = ActionInterfaceDataset(train_data_dict, context=args.context)
    val_dataset = ActionInterfaceDataset(val_data_dict, context=args.context)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle=False, num_workers=0)

    u, context = val_dataset[0]

    model = get_model(args, u_dim=len(u), a_dim=args.a_dim, c_dim=len(context), hiddens=args.hiddens, activation=args.act, 
                        lr=args.lr, L=args.lipschitz_coeff, multihead=args.multihead, model=args.model)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=None,
        filename='model-{epoch:02d}-{val_loss:.6f}',
        save_top_k=3,
        mode='min'
        )

    callbacks = [checkpoint_callback]

    # add Lipschitz regularization
    # compute layer-level Lipschitz constant
    n_layers = len(args.hiddens)+1
    if args.multihead:
        L_NN = args.lipschitz_coeff/(2*args.a_dim) # Correct for Lipschitz constant of householder transform
    else:
        L_NN = args.lipschitz_coeff/(2*np.sqrt(args.a_dim)) # Correct for Lipschitz constant of householder transform
        
    L_layer = np.power(L_NN, 1/n_layers)

    if args.lipschitz_coeff > 0.0:
        lipz = LipschitzRegularizer(lipschitz_coeff=L_layer, p=args.lipschitz_norm) 
        callbacks = callbacks + [lipz]

    logger = TensorBoardLogger(save_dir=args.default_root_dir, name=args.run_name, default_hp_metric=False)

    trainer = Trainer.from_argparse_args(args,
        logger=logger,
        callbacks=callbacks )

    pl.seed_everything(args.rnd_seed)
    
    trainer.fit(model, train_loader, val_loader)
    with open(trainer.checkpoint_callback.dirpath + '/args.yaml', 'w') as f:
        print(trainer.checkpoint_callback.dirpath)
        arg_dict = vars(args)
        yaml.dump(arg_dict, f)

def try_float(x):
    try:
        return float(x)
    except Exception as e:
        return str(x)
    

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")

    parser = Trainer.add_argparse_args(parser)
    
    parser.add_argument("--rnd_seed", default = 12345, type = int,
        help= "set random seed for experiments" )
    parser.add_argument("--run_name", default="unnamed", help="name of run in logdir")
    
    # if using dataset from D4RL
    parser.add_argument("--d4rl_dset", default='halfcheetah-expert-v2')
    parser.add_argument("--val_prop", default=0.15, help="proportion of dataset to use as validation")

    # if using custom dataset
    parser.add_argument("--train_pth", default=None,
            help="train data set json file")
    parser.add_argument("--val_pth", default=None,
            help="validation dataset")

    parser.add_argument('--a_dim', default=2, type=int,
            help="low-dimensional action size")
    parser.add_argument('--context', default='observations')
    
    # model config
    parser.add_argument('--model', default='NHT', type=str,
           help="model to run: NHT|LASER|SVD")

    parser.add_argument('--multihead', action='store_true', default=False, help='whether to multiheaded MLP')
    parser.add_argument('--hiddens', default=[256,256], type= lambda x: [int(x) for x in x.split(',')],
            help="hidden layers in encoders and decoders")
    parser.add_argument('--act', default='tanh', type=str,
            help="activation function of networks")

    # optimization config
    parser.add_argument('--lr', default=1e-3, type=float,
            help="learning rate for optimizer")
    parser.add_argument('--num-epochs', default=500, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='training batch size')
    parser.add_argument('--lipschitz_coeff', type= float, default=-1.0, 
        help="desired Lipschitz constant for decoder modules")
    parser.add_argument('--lipschitz_norm', type=lambda x: try_float(x), default=2.0,
        help="norm of Lipschitz regularizer of layers")
    
    # gpu and jit config
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
        
    args = parser.parse_args()
    print(args)
    model = main(args)
