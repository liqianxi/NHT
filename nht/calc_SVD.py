import json
from nht.utils import common_arg_parser, get_local_model_dir
from pathlib import Path
import numpy as np
import dask.array as da


def form_data_matrix(data_dict_list):
    X = np.zeros((len(data_dict_list[0]['velocity']), len(data_dict_list)))
    for i, data_pt in enumerate(data_dict_list):
        X[:,i] = data_pt['velocity']
    
    return X


def main(args):
    home = str(Path.home())
    datapath = home+args.demo_save_path
    
    train_datapath = datapath + '-train.json'
    with open(train_datapath, 'r') as f:
        train_data = json.load(f)


    X = form_data_matrix(train_data)
    p, n = X.shape
    
    if args.center:
        # Center X
        row_means = np.mean(X, axis=1)
        X = X - np.matmul(np.expand_dims(row_means,-1),np.ones((1,n)))
    else:
        row_means = np.zeros(p)
        
    X = da.from_array(X)
    print('Starting SVD computation...')
    U, S, VT = da.linalg.svd(X)
    print('SVD computation complete')
    U = np.array(U)
    S = np.array(S)
    VT = np.array(VT)

    X = np.array(X)
    
    print('Singular values:')
    print(S)
    # SVD_basis = U
    # print('full basis')
    # a = np.matmul(np.linalg.pinv(SVD_basis),np.expand_dims(X[:,0],-1))
    # r1_recon = np.matmul(SVD_basis,a)
    # print(r1_recon)
    # print(X[:,0])


    action_dim = args.action_dim
    SVD_basis = U[:,:action_dim]
    mu = np.expand_dims(row_means,-1)
    bias = np.matmul(np.linalg.pinv(SVD_basis),-1*mu)

    print('mu', mu)
    print('bias', bias)
    print('zero action', np.matmul(SVD_basis,bias)+mu)
    SVD_dict = {'U': SVD_basis.tolist(), 'mu': mu.tolist(), 'bias': bias.tolist()}
    
    model_dir = get_local_model_dir(args)
    model_name = f'SVD-{args.weight_savepath}-A{args.action_dim}.json'
    with open(f'{model_dir}/{model_name}','w+') as outfile:
        json.dump(SVD_dict, outfile)


if __name__ == '__main__':
    arg_parser = common_arg_parser()
    args = arg_parser.parse_args()
    main(args)