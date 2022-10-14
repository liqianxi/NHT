from pyparsing import ZeroOrMore
import torch.nn as nn
import torch
from torch.nn import Module

class MLP(nn.Module):
    def __init__(self, inputs, hiddens, out, activation, expmap=False, k=None): 
        super(MLP, self).__init__()
        activation = self._select_activation(activation)

        layers = [nn.Linear(inputs, hiddens[0]), activation()]
        for (in_d, out_d) in zip(hiddens[:-1], hiddens[1:]):
            
            layers = layers + [nn.Linear(in_d, out_d)]
            layers = layers + [activation()]
        layers = layers + [nn.Linear(hiddens[-1], out)]

        if expmap:
            assert k is not None
            layers = layers + [batch_ExpMap_sphere(k)]
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def _select_activation(self, act):
        if act == 'tanh':
            return nn.Tanh
        elif act == 'relu':
            return nn.ReLU
        elif act == 'sigmoid':
            return nn.Sigmoid

class MultiHeadMLP2(nn.Module):
    def __init__(self, inputs, hiddens, out, activation, k, expmap=False): 
        super(MultiHeadMLP2, self).__init__()

        assert k is not None

        activation = self._select_activation(activation)

        shared_layers = [nn.Linear(inputs, hiddens[0]), activation()]
        for (in_d, out_d) in zip(hiddens[:-1], hiddens[1:]):
            
            shared_layers = shared_layers + [nn.Linear(in_d, out_d)]
            shared_layers = shared_layers + [activation()]
            
        sub_MLPs = []
        for i in range(k):
            sub_MLP_layers = [nn.Linear(hiddens[-1], hiddens[-1]), activation(),nn.Linear(hiddens[-1], out)]
            sub_MLPs.append(nn.Sequential(*sub_MLP_layers))

        self.heads = nn.ModuleList(sub_MLPs)

        if expmap:
            self.expmap = ExpMap_sphere()
        else:
            self.expmap = None
            
        self.net = nn.Sequential(*shared_layers)

    def forward(self, x):
        shared_feature = self.net(x)
        if self.expmap is not None:
            head_outputs = [self.expmap(head(shared_feature)).unsqueeze(-1) for head in self.heads]
        else:
            head_outputs = [head(shared_feature).unsqueeze(-1) for head in self.heads]

        out = torch.cat(head_outputs,dim=-1)

        return out

    def _select_activation(self, act):
        if act == 'tanh':
            return nn.Tanh
        elif act == 'relu':
            return nn.ReLU
        elif act == 'sigmoid':
            return nn.Sigmoid

class MultiHeadMLP(nn.Module):
    def __init__(self, inputs, hiddens, out, activation, k, expmap=False): 
        super(MultiHeadMLP, self).__init__()

        assert k is not None

        activation = self._select_activation(activation)

        shared_layers = [nn.Linear(inputs, hiddens[0]), activation()]
        for (in_d, out_d) in zip(hiddens[:-1], hiddens[1:]):
            
            shared_layers = shared_layers + [nn.Linear(in_d, out_d)]
            shared_layers = shared_layers + [activation()]
            
        self.heads = nn.ModuleList([nn.Linear(hiddens[-1], out) for i in range(k)])

        if expmap:
            self.expmap = ExpMap_sphere()
        else:
            self.expmap = None
            
        self.net = nn.Sequential(*shared_layers)

    def forward(self, x):
        shared_feature = self.net(x)
        if self.expmap is not None:
            head_outputs = [self.expmap(head(shared_feature)).unsqueeze(-1) for head in self.heads]
        else:
            head_outputs = [head(shared_feature).unsqueeze(-1) for head in self.heads]

        out = torch.cat(head_outputs,dim=-1)

        return out

    def _select_activation(self, act):
        if act == 'tanh':
            return nn.Tanh
        elif act == 'relu':
            return nn.ReLU
        elif act == 'sigmoid':
            return nn.Sigmoid


class ExpMap_sphere(Module):

    def __init__(self):
        super(ExpMap_sphere, self).__init__()
        self.eps = 1e-6

    def forward(self, xi):
        
        eps = 1e-6

        batch_size = xi.shape[0]
        n = xi.shape[-1]+1

        xi_norm = torch.linalg.norm(xi,axis=-1).unsqueeze(-1)
        e1 = torch.eye(n,1).type_as(xi).squeeze()
        zero_xi_block = torch.cat([torch.zeros(batch_size,1).type_as(xi),xi],axis=-1)
        v = e1*torch.cos(xi_norm) + 1/(xi_norm+eps)*zero_xi_block*torch.sin(xi_norm)

        return v


class batch_ExpMap_sphere(Module):

    def __init__(self, k):
        super(batch_ExpMap_sphere, self).__init__()
        self.eps = 1e-6
        self.k = k
        
    def form_V_T(self, v_T):
        '''
        Forms V_T matrix, where the columns of V_T are vectors in the tangent space of the unit (n-1)-sphere at e_1

        input: v_T, batch of (n-1)k-vector
            n,   dimension of actuations (e.g. 7 for 7dof joint vel)
            k,   dimension of actions (e.g. 2 for 2dof joystick)

        output: V_T, batch of n x k matrix
        '''
        
        k = self.k
        n = int(v_T.shape[-1]/k + 1)

        zero_row = torch.zeros((1,n-1)).type_as(v_T)                      # 1 x n-1
        I = torch.eye(n-1).type_as(v_T)                                   # n-1 x n-1
        zero_augmented_I = torch.concat((zero_row, I), axis=0)            # n x n-1, first row zeros, bottom n-1 x n-1 identity

        tangent_vectors = torch.transpose(v_T.reshape(-1,k,n-1),1,2)   # batch x n-1 x k reshaped NN output

        # embed tangent vectors in ambient space of dimension n
        V_T = torch.matmul(zero_augmented_I, tangent_vectors)          # batch x n x k, first row zeros, remaining n-1 rows are reshaped NN output

        return V_T

    def forward(self, input):
        
        V_T = self.form_V_T(input)

        n = V_T.shape[1]

        norm_v_T_vec = torch.linalg.norm(V_T,axis=1)       # batch x k vector of tangent vector norms

        # compute matrices for cos term
        e1 = torch.eye(n,1).type_as(input) # n x 1 first standard basis vector [1, 0, 0, ..., 0]^T
        norm_v_T_row_vec = torch.unsqueeze(norm_v_T_vec,1)   # batch x 1 x k
        cos_norm_v_T_row_vec = torch.cos(norm_v_T_row_vec)   # batch x 1 x k  row vec where ith col is cos(norm(v_Ti))

        # compute matrices for sin term
        diag_norm_v_T = torch.diag_embed(norm_v_T_vec)       # batch x k x k diagonal matrix with tangent vector norms as diagonal 
        diag_sin_norm_v_T = torch.sin(diag_norm_v_T)
        diag_inv_norm_v_T = torch.diag_embed(1/(norm_v_T_vec+self.eps))   # batch x k x k diagonal matrix with inverse of tangent vector norms as diagonal
        diag_one_over_norm_v_T_times_sin_norm_v_T = torch.linalg.matmul(diag_sin_norm_v_T,diag_inv_norm_v_T) # batch x k x k diagonal matrix where ith diagonal is sin(norm(v_Ti))/norm(v_Ti)
        
        cos_term = torch.linalg.matmul(e1, cos_norm_v_T_row_vec)    # batch of n x k matrix with ith col of first row = cos(norm(v_Ti)), all other entries 0
        sin_term = torch.linalg.matmul(V_T, diag_one_over_norm_v_T_times_sin_norm_v_T)      # batch of n x k matrix with 
        
        V = cos_term+sin_term
        
        # if self.k == 1:  # in case you just want one vector with norm 1 instead of a matrix with columns of norm 1
        #     V = V.squeeze() 
        
        return V