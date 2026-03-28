#
# Created by : M. Ohnishi
# Created on : February 06, 2025
# 
# MIT License
# 
# Copyright (c) 2024 Masato Ohnishi at The Institute of Statistical Mathematics
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
import os.path
import numpy as np
import pandas as pd
import argparse
import torch

from phonon_e3nn.prediction import run_simulation

def clean_data(df, tol1={'gap': 10, 'kappa': 500}, tol2={'kappa': 2000},
               thred_fc2=0.1, thred_fc3=0.1):
    """ Remove materials with large phonon gap (>=10) and large thermal conductivity (>= 500), 
    and excessive thermal conductivity (>=2000). """    
    
    n0 = len(df)
    print(" Number of original data : ", len(df))
    
    ## Remove large fc2 and fc3
    df = df[(df['fc2_error'] < thred_fc2) & (df['fc3_error'] < thred_fc3)]
    df = df.reset_index(drop=True)
    if n0 != len(df):
        print(f" - Removed {n0 - len(df)} rows with fc2_error>={thred_fc2} or fc3_error>={thred_fc3}")
        n0 = len(df)
    
    ## Remove too large gap and kappa (large kappa due to absence of 4ph scattering)
    n0 = len(df)
    df = df[~((df["max_gap"] >= tol1['gap']) & (df["kp"] >= tol1['kappa']))]
    df = df.reset_index(drop=True)
    if n0 != len(df):
        print(f" - Removed {n0 - len(df)} rows with gap>={tol1['gap']} and kappa>={tol1['kappa']}")
        n0 = len(df)
    
    ## Remove excessively large kappa
    df = df[~(df["kp"] >= tol2['kappa'])]
    df = df.reset_index(drop=True)
    if n0 != len(df):
        print(f" - Removed {n0 - len(df)} rows with kappa>={tol2['kappa']}")
        n0 = len(df)

    print(" Number of available data : ", len(df))

    return df

def main(options):

    ## Set number of threads for CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if options.nprocs is not None:
        if device.type == 'cpu':
            torch.set_num_threads(options.nprocs)
    
    ## Check data file
    if os.path.exists(options.file_data) == False:
        print(" %s not found" % options.file_data)
        sys.exit()
    
    ## Prepare output directory
    os.makedirs(options.outdir, exist_ok=True)
    
    ## "num_data" is changed to observe the scaling behavior
    if options.num_data == -1:
        num_data = None
    else:
        num_data = options.num_data        # None or integer
    
    ### Load phonon data
    from phonon_e3nn.utils.utils_data import load_phonon_data
    print("\n Load %s" % options.file_data)
    df_phonon = load_phonon_data(options.file_data, target=options.target)
    
    ## Clean data
    df_phonon = clean_data(df_phonon)
    
    ## Relaxation type
    if options.which_relax == 'both':
        pass
    elif options.which_relax == 'normal':
        df_phonon = df_phonon[df_phonon['relax_type'] == 'normal']
    elif options.which_relax == 'strict':
        df_phonon = df_phonon[df_phonon['relax_type'] == 'strict']
    else:
        print("Unknown relax_type")
        sys.exit()
    
    ## Add log data
    if options.target == 'log_kp':
        df_phonon['log_kp'] = np.log10(df_phonon['kp'])
    elif options.target == 'log_kc':
        df_phonon['log_kc'] = np.log10(df_phonon['kc'])
    elif options.target == 'log_klat':
        df_phonon['log_klat'] = np.log10(df_phonon['klat'])
    
    ## Sample data
    if num_data is not None:
        df_phonon = df_phonon.sample(n=num_data, random_state=options.seed)
        print(f" Number of sampled data   : {len(df_phonon)}")
    
    ## Reset index
    df_phonon = df_phonon.reset_index(drop=True)
    
    ## alpha : weight for monotonicity penalty
    if 'cumu' in options.target.lower():
        mono_increase = True
        alpha = options.gradient_weight
    else:
        mono_increase = False
        alpha = 0.0
    
    ## Run simulation
    print()
    print('===============================')
    print('     Start simulation')
    print('===============================')
    print()
    run_simulation(
        df_phonon,
        seed=options.seed,
        target=options.target,  # 'kspec_norm' or 'kspec_mfp'
        outdir=options.outdir,
        r_max=options.r_max,
        valid_size=options.valid_size,
        test_size=options.test_size,
        random_split=False if options.random_split == 0 else True,
        batch_size=options.batch_size,
        num_epochs=options.num_epochs,
        num_epochs_limit=options.num_epochs_limit,
        patience=options.patience,
        plot_result=True,
        mono_increase=mono_increase,
        lr=options.lr,
        weight_decay=options.weight_decay,
        gamma=options.gamma,
        grad_weight=alpha,
        optimizer=options.optimizer.lower(),
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input parameters')
    
    parser.add_argument('--nprocs', dest='nprocs', type=int,
                        default=None, help="nprocs [None]")

    parser.add_argument('--file_data', dest='file_data', type=str,
                        default="../1_get/data_all.csv", help="data file name")
    
    parser.add_argument('--outdir', dest='outdir', type=str,
                        default='./out', help="output directory [./out]")
    
    parser.add_argument('--target', dest='target', type=str,
                        default="kspec_norm", 
                        help="target (kspec_norm, kspec_mfp, phdos, ...) [kspec_norm]")
    
    parser.add_argument('--which_relax', dest='which_relax', type=str,
                        default="both", 
                        help="which_relax (both, normal, strict) [both]")
    
    parser.add_argument('--num_data', dest='num_data', type=int,
                        default=None, help="output directory [./out]")
    
    parser.add_argument('--seed', dest='seed', type=int,
                        default=42, help="seed [42]")

    parser.add_argument('--r_max', dest='r_max', type=float,
                        default=4.0, help="r_max [4.0]")
    
    parser.add_argument('--valid_size', dest='valid_size', type=float,
                        default=0.1, help="valid_size [0.1]")
    parser.add_argument('--test_size', dest='test_size', type=float,
                        default=0.1, help="test_size [0.1]")
    parser.add_argument('--random_split', dest='random_split', type=int,
                        default=0, help="random split [0] (0: False, 1: True)")
    
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=16, help="batch_size [16]")
    parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                        default=3, help="num_epochs [3]")
    parser.add_argument('--num_epochs_limit', dest='num_epochs_limit', type=int,
                        default=None, help="max. limit of num_epochs [None]")
    parser.add_argument('--patience', dest='patience', type=int,
                        default=50, help="patience for loss increasing [50]")
    
    parser.add_argument('--gradient_weight', dest='gradient_weight', type=float,
                        default=0.0, 
                        help="weight for monotonicity penalty for kcumu [0.0]")
    
    ###
    ### Recommended:
    ### lr = 5.0 / N_all
    ### lr_min = 1.5 / N_all
    ### gamma = 0.95
    ###
    parser.add_argument('--lr', dest='lr', type=float,
                       default=0.001, 
                       help=(
                           "learning rate (orig: 0.005, general: 0.001, 0.0001) [0.001]. "
                           "Decrease if overfitting."))
    parser.add_argument('--lr_min', dest='lr_min', type=float,
                       default=0.0001, 
                       help="minimum learning rate [0.0001]]")
    
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=0.03, 
                        help=(
                            "weight decay (orig: 0.05, general: 0.01, 0.001) [0.03]. "
                            "Increase if overfitting."))
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.95, help="gamma [0.95]")
    
    parser.add_argument('--optimizer', dest='optimizer', type=str,
                       default='adam', 
                       help="optimizer (adam or adamw) [adam]]")
    
    args = parser.parse_args()
    
    ## Save parameters
    os.makedirs(args.outdir, exist_ok=True)
    file_params = os.path.join(args.outdir, 'params.txt')
    with open(file_params, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key:17s} : {value}\n')
    
    main(args)
    
