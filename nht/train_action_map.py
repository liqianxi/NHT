# Modified by Michael Przystupa and Kerrick Johnstonbaugh
# code is pulling snippets largely from here:
# source: https://raw.githubusercontent.com/pyro-ppl/pyro/dev/examples/vae/vae.py

from nht.utils import get_model, action_map_arg_parser, get_dsets
from nht.models.callbacks import LipschitzRegularizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from pytorch_lightning import Trainer 


def train_action_map(args):

    pl.seed_everything(args.rnd_seed)

    train_dataset, val_dataset, train_loader, val_loader = get_dsets(args)

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

    return model
    

if __name__ == '__main__':

    # parse command line arguments
    parser = action_map_arg_parser()
    args = parser.parse_args()
    print(args)

    # train action map
    model = train_action_map(args)