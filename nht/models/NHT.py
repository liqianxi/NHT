import numpy as np
import torch
import torch.nn as nn
from nht.models.MLP import MLP, MultiHeadMLP, MultiHeadMLP2
import pytorch_lightning as pl


class NHT(pl.LightningModule):
    def __init__(self,  
                u_dim=7,          
                a_dim=2,             
                c_dim=7,      
                hiddens=[256, 256],
                act = 'tanh',
                lr=1e-4,
                L=5, # not used in this class, but included as argument so it gets saved in hparams.yaml
                multihead=False
                ):

        super().__init__()
        self.save_hyperparameters()

        self.n = u_dim
        self.k = a_dim

        # create the network to predice householder vectors
        if multihead:
            self.h = MultiHeadMLP(inputs = c_dim, hiddens=hiddens, out = (self.n-1), activation=act, expmap=True, k=self.k)
        else:
            self.h = MLP(inputs = c_dim, hiddens=hiddens, out = (self.n-1)*self.k, activation=act, expmap=True, k=self.k)
            
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def householder(self, V):
        # Note, we assume the columns of V already have unit length, 
        # thanks to the exponential map on the unit sphere
        n = V.shape[-2]
        k = V.shape[-1] # number of vectors from which to construct reflections
    
        I = torch.eye(n).type_as(V)
        Q = I            # start with Q = I and keep reflecting

        for c in range(k):
            v = V[:,:,c].unsqueeze(-1)          # batch x n x 1
            vT = torch.transpose(v,1,2)         # batch x 1 x n
            vvT = torch.linalg.matmul(v,vT)     # batch x n x n
            H =  I - 2*vvT                      # batch x n x n
            Q = torch.linalg.matmul(Q, H)       # batch x n x n

        return Q 

    #def get_map(self, context):
    def forward(self, context):

        V = self.h(context)             # Calls neural network with Exp map activation on last layer. Each col of V is a unit vector v
        Q = self.householder(V)     # Computes householder reflections for each v and chains them to obtain orthonormal matrix Q
        Q_hat = Q[:,:,:self.k]      # Q_hat is first k columns of Q

        return Q_hat

    def training_step(self, train_batch, batch_idx):
        #TODO: inputs should really be formed by [conditionals, outputs], not separate value by itself
        u, context = train_batch

        Q_hat = self.forward(context)
        a_hat = torch.linalg.matmul(torch.transpose(Q_hat,1,2), u.unsqueeze(-1)) # projection
        u_hat = torch.linalg.matmul(Q_hat, a_hat).squeeze()

        loss = self.loss_fn(u_hat, u)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        u, context = val_batch

        Q_hat = self.forward(context)
        a_hat = torch.linalg.matmul(torch.transpose(Q_hat,1,2), u.unsqueeze(-1)) # projection
        u_hat = torch.linalg.matmul(Q_hat, a_hat).squeeze()

        loss = self.loss_fn(u_hat, u)

        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor =0.99)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'train_loss'
                    }
                }