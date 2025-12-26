#
# Original code: https://github.com/ninarina12/phononDoS_tutorial
# Modified by M. Ohnishi
# Modified on February 06, 2025
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
import torch

import numpy as np
import pandas as pd
import ast
import random
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde

# data visualization
import ase

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from phonon_e3nn.mpl.initialize import (set_matplot, set_axis, set_legend)

# utilities
from tqdm import tqdm

# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)

# standard formatting for plots
fontsize = 16
textsize = 14
font_family = 'sans-serif'
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

float_columns = ['kp', 'kc', 'klat', 'max_gap', 'max_phfreq', 'log_kp', 'log_kc', 'log_klat']
list_column = ['kcumu_sectinons', 'phfreq', 'phdos', 
               'kspec', 'kcumu', 'kspec_norm', 'kcumu_norm',
               'kspec_freq', 'kcumu_freq', 'kspec_norm_freq', 'kcumu_norm_freq',
               'mfp', 'log_mfp', 
               'kspec_mfp', 'kcumu_mfp', 'kspec_norm_mfp', 'kcumu_norm_mfp']

palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
datasets = ['train', 'valid', 'test']
colors = dict(zip(datasets, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])

def set_seed(seed=42):
    """ Set random seed for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_phonon_data(df, filename):
    
    df.to_csv(filename, index=False)


def load_phonon_data(filename, target='phdos', verbose=True):
    df = pd.read_csv(filename)
    return set_phonon_data(df, target=target, verbose=verbose)

def load_prediction_data(filename, verbose=False):
    ## Get target column name
    df = pd.read_csv(filename)
    targets = [col for col in df.columns if '_pred' in col]
    if len(targets) != 1:
        print(df.columns)
        raise ValueError('Prediction column not found in dataframe or too many.')
    else:
        if len(targets) > 1:
            print(targets, 'found in dataframe.')
        target = targets[0]
    # print(filename)
    df = load_phonon_data(filename, target=target, verbose=verbose)
    return df, target

def set_phonon_data(df, target='phdos', verbose=True):
    """ Set preprocess data """
    
    ## Remove invalid data
    indices_removed = {}
    for i in range(len(df)):
        for col in df.columns:
            if 'nan' in str(df[col].values[i]):
                if col not in indices_removed:
                    indices_removed[col] = []
                indices_removed[col].append(i)
    all_ids_remove = set.union(*[set(v) for v in indices_removed.values()]) if indices_removed else set()
    df_orig = df.copy()
    df = df.drop(index=all_ids_remove)
    df = df.reset_index(drop=True)
    for i, col in enumerate(indices_removed):
        if i == 0:
            print("\n Remove invalid data:")
        mpids = [df_orig['mp_id'].values[i] for i in indices_removed[col]]
        print(f" + {col} :", ", ".join(map(str, mpids)))
    if len(indices_removed) > 0:
        print()
    
    # derive formula and species columns from structure
    # structure provided as Atoms object
    if isinstance(df['structure'].values[0], str) and not df['structure'].values[0].startswith('Atoms'):
        df['structure'] = df['structure'].apply(ast.literal_eval).progress_map(lambda x: ase.Atoms.fromdict(x))
        df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
        df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    
    # convert columns to appropriate data types
    flag = False
    for col in df:
        if col in float_columns:
            df[col] = df[col].astype(float)
        elif col in list_column:
            try:
                df[col] = df[col].apply(ast.literal_eval)
            except Exception:        
                print('skip', col, '. This value may have a problem.')
                print('maybe the max value is zero for certain materials.')
        else:
            try:
                df[col] = df[col].apply(ast.literal_eval)
            except:
                pass
        
        if col == target:
            flag = True
            if isinstance(type(df[col].values[0]), float):
                df['property'] = df[col]
            else:
                df['property'] = df[col].apply(np.array)
    
    if flag == False:
        print(df.columns)
        raise ValueError(f'target column ({target}) not found in dataframe')
    
    df = df.dropna(subset=[target])
    
    # NaNチェック
    if df.isnull().values.any():
        df_nan_rows = df[df.isna().any(axis=1)]
        print(df_nan_rows)
        # raise ValueError("DataFrame contains NaN values")
        print("\n\nDataFrame contains NaN values")
    else:
        if verbose:
            print("\nNaN not found")
    
    return df


def train_valid_test_split(df, valid_size, test_size, seed=12, figname=None, random_split=False):
    
    if random_split:
        unique_mpids = df['mp_id'].unique()
        train_mpids, temp_mpids = train_test_split(
            unique_mpids, test_size=valid_size + test_size, random_state=seed)
        valid_mpids, test_mpids = train_test_split(
            temp_mpids, test_size=test_size / (valid_size + test_size), random_state=seed)
        idx_train = df[df['mp_id'].isin(train_mpids)].index.tolist()
        idx_valid = df[df['mp_id'].isin(valid_mpids)].index.tolist()
        idx_test = df[df['mp_id'].isin(test_mpids)].index.tolist()
        return idx_train, idx_valid, idx_test
    
    else:
        species = sorted(list(set(df['species'].sum())))
        
        # perform an element-balanced train/valid/test split
        print('split train/dev ...')
        dev_size = valid_size + test_size
        stats = get_element_statistics(df, species)
        idx_train, idx_dev = split_data(stats, dev_size, seed)
        
        print('split valid/test ...')
        stats_dev = get_element_statistics(df.iloc[idx_dev], species)
        idx_valid, idx_test = split_data(stats_dev, test_size/dev_size, seed)
        idx_train += df[~df.index.isin(idx_train + idx_valid + idx_test)].index.tolist()

        print('number of training examples:', len(idx_train))
        print('number of validation examples:', len(idx_valid))
        print('number of testing examples:', len(idx_test))
        print('total number of examples:', len(idx_train + idx_valid + idx_test))
        assert len(set.intersection(*map(set, [idx_train, idx_valid, idx_test]))) == 0
        
        if figname is not None:
            from phonon_e3nn.utils.plotter import plot_element_representation
            plot_element_representation(
                stats, idx_train, idx_valid, idx_test, datasets, species,
                figname=figname)
            
        return idx_train, idx_valid, idx_test


def get_element_statistics(df, species):    
    # create dictionary indexed by element names storing index of samples containing given element
    species_dict = {k: [] for k in species}
    for entry in df.itertuples():
        for specie in entry.species:
            species_dict[specie].append(entry.Index)

    # create dataframe of element statistics
    stats = pd.DataFrame({'symbol': species})
    stats['data'] = stats['symbol'].astype('object')
    for specie in species:
        stats.at[stats.index[stats['symbol'] == specie].values[0], 'data'] = species_dict[specie]
    stats['count'] = stats['data'].apply(len)
    
    return stats


def split_data(df, test_size, seed):
    # initialize output arrays
    idx_train, idx_test = [], []
    
    # remove empty examples
    df = df[df['data'].str.len()>0]
    
    # sort df in order of fewest to most examples
    df = df.sort_values('count')
    
    for _, entry in tqdm(df.iterrows(), total=len(df), bar_format=bar_format):
        df_specie = entry.to_frame().T.explode('data')
        try:
            idx_train_s, idx_test_s = train_test_split(
                df_specie['data'].values, 
                test_size=test_size,
                random_state=seed)
        except:
            pass
        else:
            # add new examples that do not exist in previous lists
            idx_train += [k for k in idx_train_s if k not in idx_train + idx_test]
            idx_test += [k for k in idx_test_s if k not in idx_train + idx_test]
    
    return idx_train, idx_test


def element_representation(x, idx):
    # get fraction of samples containing given element in dataset
    if len(x) == 0:
        return 0
    else:
        return len([k for k in x if k in idx])/len(x)


def split_subplot(ax, df, species, dataset, bottom=0., legend=False, lw=1.5):
    # plot element representation
    width = 0.4
    color = [int(colors[dataset].lstrip('#')[i:i+2], 16)/255. for i in (0,2,4)]
    bx = np.arange(len(species))
        
    ax.bar(bx, df[dataset], width, fc=color+[0.7], ec=color, lw=lw, bottom=bottom, label=dataset)
        
    ax.set_xticks(bx)
    ax.set_xticklabels(species)
    ax.tick_params(direction='in', length=0, width=1)
    ax.set_ylim(top=1.18)
    if legend:
        ax.legend(frameon=False, ncol=3, loc='upper left')

def _set_ticks_for_MFP(ax, xdat):
    
    xmin = np.min(xdat)
    xmax = np.max(xdat)
    ticks_potitions = []
    ticklabel_positions = []
    mticks_positions = []
    for n in range(-3, 5):
        
        ### major ticks
        x = 10**n
        if xmin <= x and x <= xmax:
            ticks_potitions.append(x)
            if n in [0, 2, 4]:
                ticklabel_positions.append("$10^{%d}$" % n)
            else:
                ticklabel_positions.append("")
        
        ### minor ticks
        for a in range(1, 10):
            x = a  * 10**n
            if x < xmin or x > xmax:
                continue
            mticks_positions.append(x)
    
    ax.set_xticks(ticks_potitions)
    ax.set_xticks(mticks_positions, minor=True)
    ax.set_xticklabels(ticklabel_positions)
    
    ax.tick_params(axis='x', which='major', direction='in', length=1.5, width=0.2)
    ax.tick_params(axis='x', which='minor', direction='in', length=1.0, width=0.2)


def plot_predictions_mod(df, idx, title=None, 
                         ncols=5,
                         color_true='black',
                         loss_type='mse',
                         xcol='phfreq', 
                         target='phdos', figname='fig_predictions.png',
                         fontsize=6, dpi=500, fig_width=6.0, aspect=0.35, lw=0.8,
                         wspace_main=0.4,
                         ymin_left=None, ymax_left=None,
                         ):
    
    # get quartiles
    i_mse = np.argsort(df.iloc[idx][loss_type])
    ds = df.iloc[idx].iloc[i_mse][['formula', target, target+'_pred', loss_type]].reset_index(drop=True)
    quartiles = np.quantile(ds[loss_type].values, (0.25, 0.5, 0.75, 1.))
    iq = [0] + [np.argmin(np.abs(ds[loss_type].values - k)) for k in quartiles]
    
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ### Adjust space
    ## If xdata is same for all samples
    ## => frequency is normalized by the maximum value
    same_x = True
    hspace = 0.6
    # xmax0 = np.max(df.iloc[0][xcol])
    # for i in range(min(10, len(df))):
    #     xmax = np.max(df.iloc[i][xcol])
    #     if abs(xmax - xmax0) / abs(xmax0) > 1e-3:
    #         same_x = False
    #         hspace = 1.0
    ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    nrow = min(4, int(len(idx)//ncols))
    # print(ncols, nrow)
    
    ### new version
    s = []
    for k in range(1, 5):
        population = np.arange(iq[k-1], iq[k], 1)
        sample_size = min(ncols, len(population))
        s.append(np.sort(np.random.choice(population, size=sample_size, replace=False)))
    s = np.concatenate(s)
    
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width), dpi=dpi)
    
    import matplotlib.gridspec as gridspec
    gs_main = gridspec.GridSpec(1, 5, figure=fig, wspace=wspace_main)
    i0 = 0
    gs1 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=gs_main[0,i0])
    gs2 = gridspec.GridSpecFromSubplotSpec(nrow, ncols, subplot_spec=gs_main[0,i0+1:], 
                                           hspace=hspace, wspace=0.1)
    
    ## left panel
    axl = plt.Subplot(fig, gs1[:,:])
    fig.add_subplot(axl)
    
    ## small panels
    axes_small = []
    for irow in range(nrow):
        axes_small.append([])
        for icol in range(ncols):
            axes_small[-1].append(plt.Subplot(fig, gs2[irow,icol]))
            fig.add_subplot(axes_small[irow][icol])
    
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    
    # plot quartile distribution
    y_min, y_max = ds[loss_type].min(), ds[loss_type].max()
    y = np.linspace(y_min, y_max, 501)
    
    #################################################
    nan_indices = ds[ds.isnull().any(axis=1)].index
    non_nan_indices = ds[~ds.isnull().any(axis=1)].index
    losses = ds[loss_type].values
    if len(nan_indices) > 0:
        print('\n Error: NaN found in loss values.')
        loss_ave = np.mean(losses[non_nan_indices])
        for i in nan_indices:
            losses[i] = loss_ave
            print(ds[i:i+1])
    kde = losses
    #################################################
    
    kde = gaussian_kde(losses)
    
    p = kde.pdf(y)
    y = [0] + list(y)
    p = [0] + list(p)
    
    from itertools import accumulate
    p_accumulated = np.asarray(list(accumulate(p)))
    p_accumulated /= np.max(p_accumulated)
    
    ### modify "quartiles" 
    quartiles_mod = []
    for i, k in enumerate([0.25, 0.5, 0.75, 1.]):
        inear = np.argmin(np.abs(p_accumulated - k))
        quartiles_mod.append(y[inear])
    
    axl2 = axl.twiny()
    
    axl2.plot(p, y, color="black", linewidth=lw, linestyle='--')
    axl.plot(p_accumulated, y, color="black", linewidth=lw*1.3, linestyle='-')
    
    cols = [palette[k] for k in [2,0,1,3]][::-1]
    qs =  list(quartiles_mod)[::-1] + [0]
    for i in range(len(qs)-1):
        axl.fill_between(
            [p_accumulated.min(), p_accumulated.max()],
            y1=[qs[i], qs[i]], 
            y2=[qs[i+1], qs[i+1]], 
            color=cols[i], lw=0, alpha=0.5)
        # [p.min(), p.max()], 
        # print("%2d : %.3f to %.3f " % (i, qs[i], qs[i+1]))
    
    axl2.invert_yaxis()
    axl2.set_xticks([])
    axl2.set_xlabel("Distribution (a.u.)")
    axl2.set_ylabel(loss_type.upper())
    set_axis(axl2)
    
    # axl.set_ylabel(loss_type.upper())
    # axl.ticklabel_format(axis='y', scilimits=(0,0))
    axl.invert_yaxis()
    axl.set_xlabel("Accumulation")
    axl.set_ylabel(loss_type.upper())
    set_axis(axl, xticks=0.5, mxticks=5)
    axl.grid(True, which='major', linestyle='-', c='gray', lw=0.3, axis='x')
    
    ### inverse!
    y0 = 0.
    y1 = min(qs[0], 2.5*qs[1])
    dy = y1 - y0
    
    axl.set_ylim(ymax=y0 - 0.05 * dy, ymin=y1 + 0.05 * dy)
    
    if ymin_left is not None:
        axl.set_ylim(ymax=ymin_left)
    if ymax_left is not None:
        axl.set_ylim(ymin=ymax_left)
    
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ### Set axis parameters for small panels
    if xcol == 'phfreq':
        x = df.iloc[0][xcol]
        x /= np.max(x) # normalize by the maximum value
        ## xlabel = 'Frequency (${\\rm cm^{-1}}$)'
        xlabel = '${\\rm \\omega / \\omega_{max}}$'
        xticks = 1
        mxticks = 10
        xscale = 'linear'
    elif xcol == 'log_mfp':
        
        ### ver.1 : log[MFP]
        # x = df.iloc[0][xcol]
        # xlabel = 'log[MFP (nm)]'
        # xticks = 2; mxticks = 2
        # xscale = 'linear'
        
        ### ver.2: MFP[nm]
        x = 10**np.asarray(df.iloc[0][xcol])
        xticks = None; mxticks = None
        xlabel = 'MFP (nm)'
        xscale = 'log'
        
    else:
        print('Unknown xcol:', xcol)
        print("Please add a support for this xcol.")
        
    if 'kspec' in target.lower():
        ylabel = "${\\rm \\kappa_{spec}}$"
    elif 'kcumu' in target.lower():
        ylabel = "${\\rm \\kappa_{cumu}}$"
    if '_norm' in target.lower():
        ylabel = 'Normalized ' + ylabel
    
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ## Shared y-label for all small panels
    # Add a new axis for the y-label of axs[:,1:]
    ax_y_label = fig.add_subplot(gs2[:,:])
    ax_y_label.set_ylabel(ylabel, labelpad=10)
    ax_y_label.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax_y_label.set_xticks([])
    ax_y_label.set_yticks([])
    # Remove the background and frame
    ax_y_label.patch.set_alpha(0.0)
    for spine in ax_y_label.spines.values():
        spine.set_visible(False)
    
    ## Plot predictions
    cols = np.repeat(cols[::-1], ncols)
    for k in range(4*ncols):
        
        ax = axes_small[k//ncols][k%ncols]
        
        if k >= len(s):
            break
        
        i = s[k]
        
        ax.plot(x, ds.iloc[i][target], color=color_true, linewidth=lw)
        ax.plot(x, ds.iloc[i][target+'_pred'], color=cols[k], linewidth=lw)
        
        # ### show MSE
        # mse = ds.iloc[i]['mse']
        # text = "$%.2e$" % mse
        # ax.text(0.97, 0.97, text, fontsize=5, transform=ax.transAxes, ha="right", va="top")
        
        ## all panels
        set_axis(ax, 
                 xscale=xscale, xticks=xticks, mxticks=mxticks, 
                 yscale='linear', yticks=1.0, myticks=10,
                 width=0.3, length=1.5)
        
        ###
        if xcol == 'log_mfp':
            _set_ticks_for_MFP(ax, x)
            
        ## x-axis (bottom)
        if k >= 3*ncols:
            ax.set_xlabel(xlabel, labelpad=1)
        else:
            if same_x:
                ax.tick_params(labelbottom=False)
        
        ## y-axis (left)
        ax.set_ylim([-0.05, 1.05])
        if k % ncols == 0:
            pass
        else:
            ax.tick_params(labelleft=False)

        ax.set_title(ds.iloc[i]['formula'].translate(sub), fontsize=fontsize, y=0.8)
        
        if k == len(s) - 1 or k == nrow*ncols - 1:
            break
    
    if title:
        title += " (Averaged %s: %.3f)" % (loss_type.upper(), ds[loss_type].mean())
        fig.suptitle(title, ha='center', fontsize=fontsize, y=1.0)
    
    fig.savefig(figname, dpi=dpi, bbox_inches='tight')
    print(' Output', figname)

