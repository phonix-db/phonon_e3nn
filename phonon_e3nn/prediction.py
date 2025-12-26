#
# Original code: https://github.com/ninarina12/phononDoS_tutorial
# Modified by M. Ohnishi
# Modified on February 06, 2025
# 
# MIT License
# 
# Copyright (c) 2025 Masato Ohnishi at The Institute of Statistical Mathematics
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import sys
import glob
import random

# model
import torch
import torch.nn as nn
import torch_geometric as tg
import torch_scatter
from typing import Dict, Union

# crystal structure data
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list

# data pre-processing and visualization
import numpy as np
import pandas as pd

# utilities
from tqdm import tqdm
from phonon_e3nn.utils.utils_data import set_seed, train_valid_test_split
from phonon_e3nn.utils.utils_model import Network, train
from phonon_e3nn.utils.plotter import (
    plot_lattice_parameters, plot_structure, plot_loss_history, plot_example, visualize_layers)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._check")

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

# build data
def build_data(entry, type_encoding, type_onehot, am_onehot, r_max=5.):
    """ Construct a graph from an entry in the dataframe 
    """
    symbols = list(entry.structure.symbols).copy()
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.structure, cutoff=r_max, self_interaction=True)
    
    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    
    # convert to torch tensors
    if isinstance(entry.property, float):
        target = torch.tensor(entry.property).squeeze(-1)
    else:
        target = torch.from_numpy(np.asarray(entry.property)).unsqueeze(0)
    
    data = tg.data.Data(
        pos=positions, 
        lattice=lattice, 
        symbol=symbols,
        x = am_onehot[[type_encoding[specie] for specie in symbols]],   # atomic mass (node feature)
        z=type_onehot[[type_encoding[specie] for specie in symbols]], # atom type (node attribute)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, 
        edge_len=edge_len,
        target=target,
    )
        
    return data

def add_graph_representation(df, r_max: float = 5.):
    """ Add graph representation to dataframe
    """
    ## one-hot encoding atom type and mass
    type_encoding = {}
    specie_am = []
    for Z in tqdm(range(1, 119), bar_format=bar_format):
        specie = Atom(Z)
        type_encoding[specie.symbol] = Z - 1
        specie_am.append(specie.mass)
    
    type_onehot = torch.eye(len(type_encoding))
    am_onehot = torch.diag(torch.tensor(specie_am))
    df['data'] = df.progress_apply(lambda x: build_data(x, type_encoding, type_onehot, am_onehot, r_max), axis=1)

    
# calculate average number of neighbors
def get_neighbors(df, idx):
    n = []
    for entry in df.iloc[idx].itertuples():
        N = entry.data.pos.shape[0]
        for i in range(N):
            n.append(len((entry.data.edge_index[0] == i).nonzero()))
    return np.array(n)


def _output_indices(indices, outdir='./data'):
    for key in indices.keys():
        outfile = f'{outdir}/idx_{key}.txt'
        with open(outfile, 'w') as f:
            for i in indices[key]:
                f.write(f'{i}\n')

###
class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, max_value=None, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True
            
        super().__init__(**kwargs)
        
        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)
        
        if self.irreps_out.dim == 1:
            self.output_activation = nn.Tanh()
        
        self.max_value = max_value

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.x = torch.nn.functional.relu(self.em(data.x))
        data.z = torch.nn.functional.relu(self.em(data.z))
        output = super().forward(data)
        
        if self.irreps_out.dim > 1:
            output = torch.relu(output)
        else:
            # output = nn.Identity()(output)  # for scaler output (log(kappa))
            if self.max_value is not None:
                output = self.output_activation(output) * self.max_value
            else:
                output = self.output_activation(output)
                
        # if pool_nodes was set to True, use scatter_mean to aggregate
        if self.pool == True:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)  # take mean over atoms per example
        
        if self.irreps_out.dim > 1:
            maxima, _ = torch.max(output, dim=1)
            output = output.div(maxima.unsqueeze(1))
        else:
            output = output.squeeze(-1)
        
        return output

def make_model(out_dim, r_max=5.0, n_train=None, target=None):
    """ Construct the model """
    
    if target in ['log_kp', 'log_kc', 'log_klat']:
        max_value = 5.0
    else:
        max_value = None
    
    em_dim = 64
    model = PeriodicNetwork(
        in_dim=118,                            # dimension of one-hot encoding of atom type
        em_dim=em_dim,                         # dimension of atom-type embedding
        irreps_in=str(em_dim)+"x0e",           # em_dim scalars (L=0 and even parity) on each atom to represent atom type
        irreps_out=str(out_dim)+"x0e",         # out_dim scalars (L=0 and even parity) to output
        irreps_node_attr=str(em_dim)+"x0e",    # em_dim scalars (L=0 and even parity) on each atom to represent atom type
        layers=2,                              # number of nonlinearities (number of convolutions = layers + 1)
        mul=32,                                # multiplicity of irreducible representations
        lmax=1,                                # maximum order of spherical harmonics
        max_radius=r_max,                      # cutoff radius for convolution
        num_neighbors=n_train.mean(),          # scaling factor based on the typical number of neighbors
        reduce_output=True,                    # whether or not to aggregate features of all atoms at the end
        max_value=max_value                    # maximum value of the output
    )
    return model

# class AdaptiveLoss:
#     def __init__(self, initial_a=0.001, min_a=1e-5, max_a=1.0, factor=0.1):
#         self.a = initial_a
#         self.min_a = min_a
#         self.max_a = max_a
#         self.prev_loss = None
#         self.factor = factor  # Impact on the rate of change of MSE
#    
#     def update_a(self, mse_loss):
#         if self.prev_loss is not None:
#             change = (self.prev_loss - mse_loss).item()
#             if change > 0:  # If MSE is decreasing, reduce a.
#                 self.a = max(self.min_a, self.a * (1 - self.factor * change))
#             else:  # If MSE is not decreasing, increase a.
#                 self.a = min(self.max_a, self.a * (1 + self.factor * abs(change)))
#         self.prev_loss = mse_loss
    
def monotonicity_penalty(predictions):
    # Calculate the difference of predicted values
    diffs = predictions[:, 1:] - predictions[:, :-1]
    # Impose a penalty when the difference is negative.
    penalty = torch.relu(-diffs).sum()
    # penalty = torch.mean(torch.square(torch.relu(-diffs)))
    return penalty

def custom_loss_function(predictions, targets, alpha=1.0):
    
    # calculate MSE loss
    mse_loss = torch.nn.functional.mse_loss(predictions, targets, reduction='mean')
    
    if alpha < 1e-5:
        return mse_loss
    
    # Calculate the penalty for the monotonic increasing constraint
    penalty = monotonicity_penalty(predictions)
    
    # Calculate the total loss
    scale_factor = mse_loss.detach().mean()
    total_loss = mse_loss + alpha * scale_factor * penalty
    return total_loss

# Error for the test set
def set_result(df, indices, seed, step):
    error_info = {'custom_error': {}, 'mse': {}, 'mae': {}}
    for err_kind in error_info:
        for key in indices:
            df_each = df.iloc[indices[key]]
            error_info[err_kind][key] = {
                'mean': df_each[err_kind].mean(),
                'std': df_each[err_kind].std()
            }
    df_result = pd.DataFrame()
    df_result['num_data'] = [len(df)]
    df_result['num_train'] = [len(indices['train'])]
    df_result['num_valid'] = [len(indices['valid'])]
    df_result['num_test'] = [len(indices['test'])]
    df_result['seed'] = [seed]
    df_result['epoch'] = [step]
    for err_kind, error in error_info.items():
        for key in error:
            df_result[f'{err_kind}_{key}'] = [error[key]['mean']]
            df_result[f'{err_kind}_{key}_std'] = [error[key]['std']]
    return df_result

def update_result(df1, file_result):
    if os.path.exists(file_result):
        df_large = pd.read_csv(file_result)
        exists = not df_large.merge(df1, how='inner').empty
        if not exists:
            df_large = pd.concat([df_large, df1], ignore_index=True)
        return df_large
    else:
        return df1

def run_simulation(
    df,
    target='kspec_norm', outdir='./out',
    r_max=4.0, valid_size=0.1, test_size=0.1, random_split=False,
    seed=42, batch_size=16, 
    num_epochs=1, num_epochs_limit=None, patience=50,
    lr=0.001, lr_min=None, weight_decay=0.01, gamma=0.9, 
    grad_weight=0.0, optimizer='adam',
    plot_result=False, mono_increase=False,
    ):
    
    set_seed(seed)
    
    file_result = outdir + '/result.csv'
    
    print(f'\nTarget: {target}')
    # print(f'Monotonically increasing: {mono_increase}')
    print()
    
    ## plot structure
    # i = 12
    # plot_structure(df.iloc[i]['structure'])
    
    ## plot lattice parameters
    # plot_lattice_parameters(df)
    
    ### Add graph representation
    add_graph_representation(df, r_max=r_max)
    
    ### Plot an example structure
    # i = 12 # structure index in dataframe
    # df_get = df[df['formula'] == 'S6']
    i = random.randint(0, len(df)-1)
    figname = outdir + '/fig_example.png'
    plot_example(df, i=i, label_edges=True, figname=figname)
    
    ### Training, validation, and testing datasets
    # train/valid/test split
    figname = outdir + '/fig_element_representation.png'
    idx_train, idx_valid, idx_test = train_valid_test_split(
        df, valid_size=valid_size, test_size=test_size, seed=seed,
        random_split=random_split, figname=figname)
    
    # For use with the trained model provided, the indices of the training, validation, 
    # and test sets are loaded below. These indices were generated with a specific seed using 
    # the above `train_valid_test_split` function.
    indices = {'train': idx_train, 'valid': idx_valid, 'test': idx_test}
    _output_indices(indices, outdir=outdir)
    
    # format dataloaders
    dataloader_train = tg.loader.DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size, shuffle=True)
    dataloader_valid = tg.loader.DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size)
    dataloader_test = tg.loader.DataLoader(df.iloc[idx_test]['data'].values, batch_size=batch_size)
    
    # print(type(df['data'].values[0]))
    # print(df['data'].values[0])
    # exit()
    
    n_train = get_neighbors(df, idx_train)
    n_valid = get_neighbors(df, idx_valid)
    n_test = get_neighbors(df, idx_test)
    
    # plot_example(df, n_train, n_valid, n_test, idx=12, label_edges=False)
    # exit()
    
    ### Model
    from phonon_e3nn.utils.utils_data import float_columns
    if target in float_columns:
        out_dim = 1
    else:
        out_dim = len(df.iloc[0][target])
    
    model = make_model(out_dim, r_max=r_max, n_train=n_train, target=target)
    
    ## Visualize the model
    figname = outdir + '/fig_model.png'
    visualize_layers(model, figname=figname)

    ### Training
    # The model is trained using a mean-absolute error loss function with an Adam or AdamW optimizer.
    if optimizer.lower() == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Unsupported optimizer:', optimizer)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    
    loss_funcs = {
        'mse': torch.nn.MSELoss(),
        'mae': torch.nn.L1Loss(),
        'custom': custom_loss_function,
        'grad_weight': grad_weight
        # 'adaptive': AdaptiveLoss(),
    }
    
    ## Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('\ntorch device:' , device)
    
    line = f'{outdir}/model*.torch'
    fns = glob.glob(line)
    if len(fns) > 0:
        print(f'\nFound {len(fns)} models in {outdir}')
        print(f'Use', fns[0])
        run_name = fns[0].replace('.torch', '')
    else:    
        run_name = f'{outdir}/model_{target}_ndata{len(df)}_seed{seed}'
    
    file_model = run_name + '.torch'
    # file_best_model = run_name + '_best.torch'
    # file_history = f'{outdir}/history.csv'
    # print(run_name)
    
    ### Training
    model.pool = True
    train(model, opt, 
          dataloader_train, dataloader_valid, dataloader_test,
          loss_funcs,
          file_model=file_model,
          mono_increase=mono_increase,
          num_epochs=num_epochs, 
          num_epochs_limit=num_epochs_limit,
          patience=patience,
          scheduler=scheduler, device=device,
          lr_min=lr_min,
          outdir=outdir)
    
    ## Ensure target tensor is reshaped correctly
    for d in dataloader_train:
        if len(d.target.shape) == 1:
            d.target = d.target.unsqueeze(-1)
    for d in dataloader_valid:
        if len(d.target.shape) == 1:
            d.target = d.target.unsqueeze(-1)
    for d in dataloader_test:
        if len(d.target.shape) == 1:
            d.target = d.target.unsqueeze(-1)
    
    # Check for NaN in input data
    # def check_for_nan(tensor, tensor_name):
    #     if torch.isnan(tensor).any():
    #         print(f"NaN detected in {tensor_name}")
    #         return True
    #     return False
    
    # for dl in [dataloader_train, dataloader_valid, dataloader_test]:
    #     for d in dl:
    #         if check_for_nan(d.pos, "d.pos") or check_for_nan(d.x, "d.x") or check_for_nan(d.z, "d.z"):
    #             raise ValueError("NaN detected in input data")
    
    # # Debugging: Check for NaN in model layers
    # def debug_model_output(model, data):
    #     for name, layer in model.named_children():
    #         if isinstance(layer, torch.nn.ModuleList):
    #             for sub_layer in layer:
    #                 data = sub_layer(data)
    #                 if check_for_nan(data, f"layer {name}"):
    #                     raise ValueError(f"NaN detected in layer {name}")
    #         else:
    #             data = layer(data)
    #             if check_for_nan(data, f"layer {name}"):
    #                 raise ValueError(f"NaN detected in layer {name}")
    #     return data
    
    ## load pre-trained model and plot its training history
    history = torch.load(file_model, weights_only=True, map_location=device)['history']
    steps = [d['total_step'] for d in history]
    loss_train = [d['train']['loss'] for d in history]
    loss_valid = [d['valid']['loss'] for d in history]
    figname = outdir + '/fig_loss.png'
    plot_loss_history(steps, loss_train, loss_valid, figname=figname)
    
    ## load pre-trained model
    history = torch.load(file_model, weights_only=True, map_location=device)['history']
    model.load_state_dict(torch.load(file_model, weights_only=True, map_location=device)['best'])
    model.pool = True

    dataloader = tg.loader.DataLoader(df['data'].values, batch_size=batch_size)
    df['custom_error'] = 0.
    df['mse'] = 0.
    df['mae'] = 0.
    df[target+'_pred'] = np.empty((len(df), 0)).tolist()
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        i0 = 0
        for i, d in enumerate(dataloader):
            
            d.to(device)
            output = model(d)
            
            if len(output.shape) == 1 and len(d.target.shape) == 2:
                output = output.unsqueeze(-1)
            
            loss_mse = nn.functional.mse_loss(output, d.target, reduction='none').mean(dim=-1).cpu().numpy()
            loss_mae = nn.functional.l1_loss(output, d.target, reduction='none').mean(dim=-1).cpu().numpy()
            
            if d.target.shape[1] == 1:
                df.loc[i0:i0 + len(d.target) - 1, target+'_pred'] = output.cpu().numpy()
            else:
                df.loc[i0:i0 + len(d.target) - 1, target+'_pred'] = [[k] for k in output.cpu().numpy()]
            
            df.loc[i0:i0 + len(d.target) - 1, 'mse'] = loss_mse
            df.loc[i0:i0 + len(d.target) - 1, 'mae'] = loss_mae
            i0 += len(d.target)
            
    if isinstance(df[target+'_pred'].values[0], float):
        target_type = float
    else:
        target_type = list
        df[target+'_pred'] = df[target+'_pred'].map(lambda x: x[0])
    
    ###
    df_result_now = set_result(df, {'train': idx_train, 'valid': idx_valid, 'test': idx_test}, seed, steps[-1])
    df_result = update_result(df_result_now, file_result)
    df_result.to_csv(file_result, index=False)
    print()
    print(df_result)
    
    if plot_result:
        
        if '_freq' in target.lower():
            xcol = 'phfreq'
        elif '_mfp' in target.lower():
            xcol = 'log_mfp'
        elif target in float_columns:
            xcol = None
        else:
            print('\n Unsupported target:', target, '. Defined xcol and xscale.')
            xcol = None
        
        ### Save data with predictions
        df_pred = pd.DataFrame()
        copy_columns = ['mp_id', 'formula', 'structure', 'max_phfreq', xcol, target, target+'_pred', 'mse']
        for col in copy_columns:
            
            if df.get(col) is None and col != target+'_pred':
                continue
            
            if col in ['mp_id', 'formula', 'structure']:
                df_pred[col] = df[col].values
            else:
                if col == target + '_pred':
                    if target_type == list:
                        df_pred[target+'_pred'] = (
                            df[target+'_pred'].apply(
                                lambda x: x.tolist() 
                                if isinstance(x, np.ndarray) 
                                else x)
                        )
                    elif target_type == float:
                        df_pred[target+'_pred'] = df[target+'_pred'].copy()
                else:
                    df_pred[col] = df[col].values.tolist()
        
        df_pred['kind'] = 'kind'
        df_pred.iloc[idx_train, df_pred.columns.get_loc('kind')] = 'train'
        df_pred.iloc[idx_valid, df_pred.columns.get_loc('kind')] = 'valid'
        df_pred.iloc[idx_test, df_pred.columns.get_loc('kind')] = 'test'
        outfile = outdir + '/data_pred.csv'
        df_pred.to_csv(outfile, index=False)
        print(f'\nSaved data with predictions: {outfile}')
        
        ### Plot predictions
        titles = {'train': 'Training', 'valid': 'Validation', 'test': 'Testing'}
        if target_type == list:
            from phonon_e3nn.utils.utils_data import plot_predictions_mod
            for key in indices:
                figname = outdir + f'/fig_{key}.png'
                plot_predictions_mod(
                    df, indices[key], 
                    loss_type='mae',
                    title=titles[key], 
                    xcol=xcol, target=target, figname=figname)
        else:
            from phonon_e3nn.utils.plotter import plot_prediction_parity
            figname = outdir + '/fig_parity.png'
            plot_prediction_parity(df, indices, target=target, figname=figname)

