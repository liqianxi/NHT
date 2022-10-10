import numpy as np
import torch
import torch.nn as nn
from latent_robotic_actions.models.networks import MLP, HypernetworkLinear
from latent_robotic_actions.models.losses import get_mmd_loss, log_prob_gauss, norm_error
import pytorch_lightning as pl
from latent_robotic_actions.models.human_prior_losses import proportionality, consistency, reversibility
from latent_robotic_actions.models.torch_kinematics import TorchKinovaKinematics, TorchPlanarFiveDOFKinematics


class NHT(pl.LightningModule):
    def __init__(self,  inputs=None,         # in CAE, inputs=['thetas','velocity'], for NHT there is no encoder (uses actuation projection instead)
                        outputs=7,           # outputs=['velocity']
                        z_dim=2,             # dimensionality of low-dim actions, k
                        cond_dims = 7,       # cond_inp=['thetas']
                        hiddens=[256, 256],
                        act = 'tanh',
                        lr=1e-4,
                        ):

        super().__init__()
        self.save_hyperparameters()


        # create the network to predict tangent vectors
        #self.h = MLP(inputs= z_dim, outputs = outputs-1, hyper_inp=cond_dims, hiddens=hiddens, act=act)
        self.n = outputs
        self.k = z_dim

        # create the network to predice householder vectors
        self.decoder = MLP(inputs = cond_dims, hiddens=hiddens, out = (self.n-1)*self.k, activation=act, expmap=True, k=self.k)

        self.lr = lr

        self.I_k_Zero_block = None
        
    

    def householder(self, V):
        # Note, we assume the columns of V already have unit length, 
        # thanks to the exponential map to the unit sphere
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

    def get_map(self, obs):

        V = self.decoder(obs)
        Q = self.householder(V)
        Q_hat = torch.linalg.matmul(Q, self.get_I_k_Zero_block())

        return Q_hat

    def decode(self, z, cond_inpt):
        '''
        Gives output (qdot) given conditional input and action
        '''
        Q_hat = self.get_map(cond_inpt)
        a = z.unsqueeze(-1)
        
        qdot = torch.linalg.matmul(Q_hat,a)
        
        return qdot.squeeze()


    def forward(self, qdot, cond_inp = None):
        '''
        Gives qdot_hat and corresponding low dimenional action
        '''

        Q_hat = self.get_map(cond_inp)
        qdot = qdot.unsqueeze(-1)  # make column vector for matrix vector multiplication
        a_hat = torch.linalg.matmul(torch.transpose(Q_hat,1,2), qdot)
        qdot_hat = torch.linalg.matmul(Q_hat, a_hat).squeeze()

        return qdot_hat, a_hat


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor =0.99)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'train_loss'
                    }
                }


    def training_step(self, train_batch, batch_idx):
        #TODO: inputs should really be formed by [conditionals, outputs], not separate value by itself
        input, qdot, obs = train_batch

        qdot_hat, a_hat = self.forward(qdot, cond_inp=obs)

        loss = log_prob_gauss(qdot_hat, qdot, torch.ones_like(qdot))

        self.log('train_loss', loss)

        return loss


    def validation_step(self, val_batch, batch_idx):
        input, qdot, obs  = val_batch

        qdot_hat, a_hat = self.forward(qdot, cond_inp=obs)

        loss = log_prob_gauss(qdot_hat, qdot, torch.ones_like(qdot))

        self.log('val_loss', loss)
        return loss

    def get_I_k_Zero_block(self):
        # work around to get constant matrix on gpu
        if self.I_k_Zero_block is not None:
            return self.I_k_Zero_block

        # form block matrix to select first k columns of Q
        I_k = torch.eye(self.k, device=self.device)
        Zero = torch.zeros(self.n-self.k, self.k, device=self.device)
        self.I_k_Zero_block = torch.concat((I_k,Zero), axis=0)
        return self.I_k_Zero_block
    