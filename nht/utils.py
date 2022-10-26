import datetime
from pathlib import Path
import os
from pytorch_lightning import Trainer 
import json
from torch.utils.data import Dataset
import torch
import random
import gym
from torch.utils.data import DataLoader
import d4rl

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def try_float(x):
    try:
        return float(x)
    except Exception as e:
        return str(x)

def action_map_arg_parser():
  parser = arg_parser()
  parser = Trainer.add_argparse_args(parser)

  parser.add_argument("--rnd_seed", default = 12345, type = int,
      help= "set random seed for experiments" )
  parser.add_argument("--run_name", default="unnamed", help="name of run in logdir")

  # if using dataset from D4RL
  parser.add_argument("--d4rl_dset", default=None)
  parser.add_argument("--dataset_prop", type=lambda x: try_float(x), default=1.0, help="proportion of overall dataset to use")
  parser.add_argument("--dataset_transitions", type=int, default=None, help="number of transitions of dataset to use")
  parser.add_argument("--val_prop", type=lambda x: try_float(x), default=0.15, help="proportion of used dataset to be validation")

  # if using custom dataset
  parser.add_argument("--train_pth", default=None,
          help="train data set json file")
  parser.add_argument("--val_pth", default=None,
          help="validation dataset")

  # config context and low-dimensional action space of NHT
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

  return parser

import numpy as np
from tqdm import tqdm
import pickle
import pathlib
def project_d4rl_actions(NHT_path, d4rl_dset, prop=1.0, attempt_load=True):

  env = gym.make(d4rl_dset)
  data_dict = env.get_dataset()
  N_examples = int(len(data_dict['actions'])*prop)

  if attempt_load:
    try:
      with open(f"./.datasets/{d4rl_dset}_projected.pickle","rb") as f:
        data_dict = pickle.load(f)

      assert len(data_dict['actions']) == N_examples
      return data_dict
    except:
      print("Projecting data...")

  Q_func = load_interface('NHT', model_path = NHT_path)

  print(f"Total training examples: {N_examples}")

  projected_actions = np.zeros((N_examples,2))
  
  for idx in tqdm(range(N_examples)):
    # if idx % 1000 == 0:
    #   print(idx)
    u = torch.from_numpy(data_dict['actions'][idx]).unsqueeze(0)
    c = torch.from_numpy(data_dict['observations'][idx]).unsqueeze(0)
    Q_hat = Q_func(c)
    a_hat = torch.linalg.matmul(torch.transpose(Q_hat,1,2), u.unsqueeze(-1)) # projection
    projected_actions[idx] = a_hat.squeeze().detach().numpy()
  
  data_dict['actions'] = projected_actions

  # make dir if it doesn't exist
  data_dir = pathlib.Path("./.datasets")
  data_dir.mkdir(exist_ok=True)

  # save data
  with open(f"./.datasets/{d4rl_dset}_projected.pickle","wb") as f:
    pickle.dump(data_dict, f)
  
  return data_dict

def get_dsets(args):
  if args.d4rl_dset is not None:
    env = gym.make(args.d4rl_dset)
    data_dict = env.get_dataset()
    train_data_dict, val_data_dict = split_dset(data_dict, prop=args.dataset_prop, transitions=args.dataset_transitions, context=args.context, val_prop=args.val_prop)
  else:
    raise NotImplementedError

  #create datasets
  train_dataset = ActionInterfaceDataset(train_data_dict, context=args.context)
  val_dataset = ActionInterfaceDataset(val_data_dict, context=args.context)

  train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle=False, num_workers=0)

  return train_dataset, val_dataset, train_loader, val_loader

class ActionInterfaceDataset(Dataset):
  def __init__(self, data_dict = None, context='observations'):
    
    self.examples = []
    
    for action, obs in zip(data_dict['actions'], data_dict[context]):
      self.examples.append(dict(u = torch.tensor(action), context = torch.tensor(obs)))

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx):
    example = self.examples[idx]
    return example['u'], example['context']


def split_dset(data_dict, prop, transitions, context, val_prop):

  N = len(data_dict['actions'])


  indices = [*range(N)]
  random.shuffle(indices)
  
  if transitions is not None:
    N = transitions
  else:
    N = int(N*prop)
  
  indices = indices[:N] # take first N indices

  N_val = int(val_prop*N)
  N_train = N-N_val

  train_indices = indices[:N_train]
  val_indices = indices[N_train:]

  assert len(val_indices) == N_val

  train_actions = data_dict['actions'][train_indices]
  train_context = data_dict[context][train_indices]

  train_data_dict = {'actions': train_actions, context: train_context}

  val_actions = data_dict['actions'][val_indices]
  val_context = data_dict[context][val_indices]

  val_data_dict = {'actions': val_actions, context: val_context}

  return train_data_dict, val_data_dict


from nht.models.NHT import NHT
def get_model(args, u_dim, a_dim, c_dim, hiddens, activation, lr, L, multihead, model):
    
    if model == 'NHT':
      return NHT(u_dim=u_dim, a_dim=a_dim, c_dim=c_dim, hiddens=hiddens, act=activation, L=L, multihead=multihead, lr=lr)

def load_model(model):
    if model == 'NHT':
      return NHT

import glob
import yaml

def load_interface(model_type, model_path):

    folder = f'{model_path}/checkpoints/'
    #print(folder)
    models = glob.glob(folder + '/*.ckpt')
    #print(models)
    
    best = sorted(models, key= lambda x: float(x.split('val_loss=')[1].split('.ckpt')[0]), reverse=False)[0]
    with open(folder + 'args.yaml', 'r') as f:
        exp_args = yaml.safe_load(f)
    #print(exp_args['model'])

    model = load_model(model_type).load_from_checkpoint(checkpoint_path=best)
        
    #print(model)

    return model