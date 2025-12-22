#
# Created by M. Ohnishi
# Created on February 06, 2025
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
import math
import numpy as np
import pandas as pd
import argparse
import glob

from scipy.stats import bootstrap
# from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from phonon_e3nn.mpl.initialize import (set_matplot, set_axis, set_legend)

from phonon_e3nn.utils.utils_data import load_prediction_data
from phonon_e3nn.utils.plotter import plot_prediction_single
from phonon_e3nn.utils.scaling import plot_scaling_law, scaling_function, modify_ticklabels_log

def get_confidence_interval(values):
    bst = bootstrap((values,), np.mean, confidence_level=0.9, random_state=42)
    low  = bst.confidence_interval.low
    high = bst.confidence_interval.high
    return low, high

def _get_error_data(filenames, col_error='mae_valid'):
    dfs = []
    for fn in filenames:
        df = pd.read_csv(fn)
        # print(fn)
        if len(df) > 0:
            imin = np.argmin(df[col_error].values)
            dfs.append(df[imin:imin+1])
    df = pd.concat(dfs).reset_index(drop=True)
    return df

def plot_scaling(
    ax, df, xcol='num_data', ycol='mse', ratio_train=0.8,
    xlabel='Training data size', ylabel='MSE',
    lw=0.8, ms=2.5):
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    xdat = df[xcol].values * ratio_train
    try:
        ydat = df[f'{ycol}_mean'].values
    except:
        ydat = df[ycol].values
    yerr = [df[f'{ycol}_low'].values, df[f'{ycol}_high'].values]
    
    ### ver.1: with residual error
    # popt = _plot_fitting(ax, xdat, ydat)
    # yscale = 'linear'
    
    # ### ver.2: linear scaling
    popt = plot_scaling_law(ax, xdat, ydat, show_line=False, p0=[0.1, -10.0])
    yscale = 'log'
    
    ax.errorbar(xdat, ydat, yerr=yerr, linestyle='None', lw=lw, marker='o', markersize=ms,
                mfc='none', mew=lw, capsize=2, color='blue')
    
    ### ticks
    if np.max(ydat) - np.min(ydat) < 0.03:
        yticks = 0.005
        myticks = 2
    else:
        yticks = None
        myticks = None
    
    set_axis(ax, xscale='log', yscale=yscale, yticks=yticks, myticks=myticks)
    
    return popt


def make_frame(fontsize=7, fig_width=None, aspect=None, 
               ws_main=0.6, hs_right=0.4, frac_left=0.7,
               plot_scaling_only=False):
    
    import matplotlib.gridspec as gridspec
    
    set_matplot(fontsize=fontsize)
    fig = plt.figure(figsize=(fig_width, aspect*fig_width))
    plt.subplots_adjust(wspace=ws_main, hspace=0.1)
    
    if plot_scaling_only:
        ax = plt.subplot()
        return fig, ax, None
    
    gs_main = gridspec.GridSpec(1, 5, figure=fig)
    
    ### Left panel
    gs_left = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs_main[:int(frac_left*10/2)])
    axl = plt.Subplot(fig, gs_left[0])
    fig.add_subplot(axl)
    
    ### Right panel
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_main[int(frac_left*10/2):], hspace=hs_right)
    axes_right = []
    for ii in range(3):
        ax = plt.Subplot(fig, gs_right[ii])
        fig.add_subplot(ax)
        axes_right.append(ax)
    
    ### Add labels
    # axl.text(0.00, 1.03, '(a)', fontsize=7, transform=axl.transAxes, ha="left", va="bottom")
    # axl.text(1.05, 1.03, '(b)', fontsize=7, transform=axl.transAxes, ha="left", va="bottom")
    
    return fig, axl, axes_right
    
def _get_scaling_data(min_epoch=100, min_ns=0):
    
    kinds_of_loss = ['mse', 'mae']
    
    dump = []
    for n in [-1] + [j for j in range(min_ns, 6000)]:
        
        line = f'out_N{str(n)}/seed*/result.csv'
        fns = glob.glob(line)
        if len(fns) == 0:
            continue
        
        df = _get_error_data(fns, col_error='mae_valid')
        
        cols_shown = []
        for col in df.columns:
            if col.startswith('num_') or col.startswith('mse_') or col.startswith('mae_'):
                cols_shown.append(col)
            elif col in ['seed', 'epoch', 'kind']:
                cols_shown.append(col)
        
        df = df.sort_values('seed')
        print(df[cols_shown])
        
        ### filter by min_epoch
        if min_epoch is not None:
            df = df[df['epoch'] >= min_epoch]
        
        if len(df) == 0:
            continue
        
        error_info = {key: {} for key in kinds_of_loss}
        if len(df) == 1:
            num_data = df['num_data'].values[0]
            num_train = df['num_train'].values[0]
            num_valid = df['num_valid'].values[0]
            num_test = df['num_test'].values[0]
            for kind in kinds_of_loss:
                error_info[kind]['mean'] = df[f'{kind}_test'].values[0]
                error_info[kind]['low'] = 0.0
                error_info[kind]['high'] = 0.0
        else:
            num_data = int(df['num_data'].mean())
            num_train = int(df['num_train'].mean())
            num_valid = int(df['num_valid'].mean())
            num_test = int(df['num_test'].mean())
            for kind in kinds_of_loss:
                error_info[kind]['mean'] = df[f'{kind}_test'].mean()
                low, high = get_confidence_interval(df[f'{kind}_test'])
                error_info[kind]['low'] = error_info[kind]['mean'] - low
                error_info[kind]['high'] = high - error_info[kind]['mean']
        
        dump.append([num_data, num_train, num_valid, num_test, len(df)])
        for kind in kinds_of_loss:
            dump[-1].append(error_info[kind]['mean'])
            dump[-1].append(error_info[kind]['low'])
            dump[-1].append(error_info[kind]['high'])
    
    new_columns = ['num_data', 'num_train', 'num_valid', 'num_test', 'ensemble']
    for kind in kinds_of_loss:
        new_columns.append(f'{kind}_mean')
        new_columns.append(f'{kind}_low')
        new_columns.append(f'{kind}_high')
    
    df_scale = pd.DataFrame(dump, columns=new_columns)
    df_scale = df_scale.sort_values('num_data').reset_index(drop=True)
    
    print(df_scale)
    
    return df_scale

def _add_loss(df, loss_type=None):
    
    target = [col for col in df.columns if '_pred' in col][0].replace('_pred', '')
    
    error_info = {'mae': [], 'mse': []}
    for i in range(len(df)):
        value1 = np.asarray(df.iloc[i][target])
        value2 = np.asarray(df.iloc[i][target+"_pred"])
        try:
            mae_each = np.abs(value1 - value2).mean()
        except:
            mae_each = np.nan
        error_info['mae'].append(mae_each)
        
        try:
            mse_each = np.power(value1 - value2, 2).mean()
        except:
            mse_each = np.nan
        error_info['mse'].append(mse_each)
    
    diff = df['mse'].values - np.asarray(error_info['mse'])
    if np.max(np.abs(diff)) > 1e-3:
        raise ValueError("Inconsistent data")
        
    df['mae'] = error_info['mae']
    df['mse'] = error_info['mse']

def _get_prediction_data(line, kind='test', num_files=None, loss_type='mae'):
    
    filenames = glob.glob(line)
    dfs = []
    counter = 0
    for fn in filenames:
        
        df_each, _ = load_prediction_data(fn, verbose=False)
        
        if loss_type not in df_each.columns:
            _add_loss(df_each, loss_type=loss_type)
        
        dfs.append(df_each)
        
        counter += 1
        if num_files is not None and counter >= num_files:
            break
        
    if len(dfs) == 0:
        raise ValueError("No data found for %s" % line)
    
    df_pred = pd.concat(dfs).reset_index(drop=True)
    try:
        df_pred = df_pred.dropna(subset=[loss_type])
    except:
        pass
    
    if kind is not None:
        df_pred = df_pred[df_pred['kind'] == kind]
    
    return df_pred.reset_index(drop=True)

def generate_logscale_list(v0, v1):
    values = []
    power = -2
    while True:
        base = 10 ** power
        for i in range(2, 10):
            num = i * base
            if num > v1:
                return values
            if num >= v0:
                values.append(num)
        power += 1
        
def modify_xaxis_for_mfp(ax, log_values):
    
    log_vmin = np.min(log_values)
    log_vmax = np.max(log_values)
    log_dv = log_vmax - log_vmin
    log_v0 = log_vmin - 0.05 * log_dv
    log_v1 = log_vmax + 0.05 * log_dv
    ax.set_xlim(log_v0, log_v1)
    set_axis(ax, xscale='linear')
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    ax2 = ax.twiny()
    v0 = np.power(10, log_v0)
    v1 = np.power(10, log_v1)
    ax2.set_xlim(v0, v1)
    set_axis(ax2, xscale='log')
    ax2.tick_params(
        labelleft=False, left=False, 
        labelright=False, right=False, 
        which='both')
    
    ### manual ticks!!!
    major_list = []
    major_ticklabels = []
    for i in range(math.ceil(log_v0), math.floor(log_v1) + 1):
        major_list.append(10**i)
        if i in [0, 2, 4]:
            major_ticklabels.append("${\\rm 10^{%d}}$" % i)
        else:
            major_ticklabels.append("")
    minor_list = generate_logscale_list(v0, v1)
    
    ax2.set_xticks(major_list, minor=False)
    ax2.set_xticklabels(major_ticklabels, minor=False)
    ax2.set_xticks(minor_list, minor=True)
    ax2.tick_params(axis='x', which='minor', length=1.3, width=0.2)
    ax2.tick_params(axis='x', which='major', length=1.8, width=0.3)
    
    return ax2

def main(options):
    
    loss_type = 'mae'
    if options.ylabel is None:
        ylabel = loss_type.upper()
    else:
        ylabel = options.ylabel

    cmap = plt.get_cmap("tab10")
    
    fig, axl, axes_right = make_frame(
        fig_width=3.5, aspect=0.8, ws_main=1.5, hs_right=0.45, frac_left=0.6)
    
    ### Get scaling data and save it
    df_scale = _get_scaling_data(min_epoch=options.min_epoch, min_ns=options.min_ns)
    df_scale.to_csv(options.outfile, index=False)
    print(" Output", options.outfile)
    
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ### Scaling law
    # print(df_scale.columns)
    xcol = 'num_data'
    ratio_train = 0.8
    popt = plot_scaling(
        axl, df_scale, 
        xcol=xcol, ycol=loss_type, 
        ratio_train=ratio_train, 
        ylabel=ylabel)
    
    ### Set axis range
    def _set_axis_range(ax, values, which='x', scale='linear', buffer=0.1):
        v0 = np.min(values)
        v1 = np.max(values)
        if scale == 'log':
            v0 = np.log10(v0)
            v1 = np.log10(v1)
        vmin = v0 - buffer * (v1 - v0)
        vmax = v1 + buffer * (v1 - v0)
        if scale == 'log':
            vmin = np.power(10, vmin)
            vmax = np.power(10, vmax)
        # if which == 'x':
        #     ax.set_xlim(vmin, vmax)
        # elif which == 'y':
        #     ax.set_ylim(vmin, vmax)
        return [vmin, vmax]
    
    ### Set x-axis range
    xlim = _set_axis_range(
        fig.axes[0], df_scale[xcol].values * ratio_train, which='x', scale='log', buffer=0.1)
    xlim[1] = 2e5
    axl.set_xlim(xlim)
    
    ### draw the line for scaling law
    xfit = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 51)
    yfit = scaling_function(xfit, *popt)
    axl.plot(xfit, yfit, color='grey', linestyle='-', alpha=0.5, lw=5, zorder=1)
    
    # set_axis(axl, yscale='linear', xscale='log', 
    #          yticks=options.yticks, myticks=options.myticks)
    set_axis(axl, yscale='log', xscale='log')
    
    ### Set major ticks (y-axis)
    modify_ticklabels_log(
        axl, np.min(yfit), np.max(yfit) * 2.0,
        label_positions=[0.01, 0.1, 1.0], which='y', minor=False)
    
    #### Manual setting for minor ticks ###################################
    ### Set minor ticks (y-axis) 
    if options.target == 'kspec':
        label_pos = [i*0.01 for i in range(5, 10)]
        axl.set_yticks(label_pos, minor=True)
        axl.set_yticklabels(label_pos, minor=True)
    elif options.target == 'kcumu':
        tick_positions = []
        tick_lables = []
        for i in range(3, 16):
            tick_positions.append(0.01 * i)
            if str(i) in options.mytick_positions.split(':'):
                tick_lables.append(f'{0.01 * i:.2f}')
            else:
                tick_lables.append('')
        axl.set_yticks(tick_positions, minor=True)
        axl.set_yticklabels(tick_lables, minor=True)
    #######################################################################
    
    if options.ymin is not None:
        axl.set_ylim(ymin=options.ymin)
    if options.ymax is not None:
        axl.set_ylim(ymax=options.ymax)
    
    target = options.target
    
    ############################################################
    ## Prediction examples (right column)
    ycol = loss_type+'_mean'
    
    if len(df_scale) <= 4:
        n0, n1 = 0, 1
    else:
        n0, n1 = 0, 2
    
    col_indices = [3, 1, 0]
    n2 = -1
    for i, idx_num in enumerate([n0, n1, n2]):
        
        ax = axes_right[i]
    
        ## marker data (# of samples)
        # Ns = df_scale.iloc[idx_num][xcol]
        Ntot = df_scale.iloc[idx_num]['num_data']
        error_mean = df_scale.iloc[idx_num][ycol]
        
        ## read prediction data for x = Ns
        if idx_num == -1:
            nominal_ndat = '-1'
        else:
            if Ntot < 900:
                n = round(df_scale.iloc[idx_num][xcol], -1)
            elif Ntot < 9000:
                n = round(df_scale.iloc[idx_num][xcol], -2)
            else:
                n = round(df_scale.iloc[idx_num][xcol], -3)
            
            nominal_ndat = n
        
        line = f"./out_N{int(nominal_ndat)}/seed*/data_pred.csv"
        fns = glob.glob(line)
        if len(fns) == 0:
            continue
        
        df_pred = _get_prediction_data(line, num_files=None, loss_type=loss_type)
        df_pred = df_pred.sort_values(loss_type).reset_index(drop=True)
        
        if i == 0:
            col = cmap(3)
            #################
            # break
            #################

        # col = cmap(1) if i == 1 else cmap(0)
        col = cmap(col_indices[i])
        
        ## plot an example
        isort = np.argsort(abs(df_pred[loss_type] - error_mean))
        try:
            inear = isort[0]
        except:
            inear = isort[2]
        
        target = [col for col in df_pred.columns if '_pred' in col][0].replace('_pred', '')
        
        ## find a visually-nice result (ymin < 0.2)
        if target.startswith('kcumu'):
            count = 0
            while True:
                ydat = np.asarray(df_pred[target].values[isort[count]])
                if np.min(ydat) < 0.2:
                    inear = isort[count]
                    break
                count += 1
        
        plot_prediction_single(
            axes_right[i], df_pred.iloc[inear], lw=1.0, col_pred=col, 
            loss_type=loss_type)
        
        if 'kspec' in target.lower() and i == 2:
            set_legend(ax, fs=5, loc='upper right', alpha=0.0, lw=0.0)
        elif 'kcumu' in target.lower() and i == 0:
            set_legend(ax, fs=5, loc='center right', alpha=0.0, lw=0.0,
                       loc2=[1.0, 0.4])
        
        ## Best
        if 'kspec' in target.lower():
            y = 0.2
        else:
            y = 0.1
        text = "$\\rm N_{all} = %d$" % (Ntot)
        ax.text(0.95, y, text, fontsize=5,
                transform=ax.transAxes, 
                ha="right", va="bottom")
    
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    n = len(axes_right)
    if 'kcumu' in target.lower():
        xlabel = "MFP (nm)"
        for i in range(n):
            ax = axes_right[i]
            ax2 = modify_xaxis_for_mfp(ax, df_pred['log_mfp'].values[0])
            ax.set_xlabel(None)
            if i < n - 1:
                ax2.set_xlabel(None)
                ax2.set_xticklabels([]) 
            else:
                ax2.xaxis.tick_bottom()
                ax2.xaxis.set_label_position('bottom')
                ax2.xaxis.set_ticks_position('both')
                ax2.set_xlabel(xlabel)
    elif 'kspec' in target.lower():
        xlabel = "${\\rm \\omega / \\omega_{max}}$"
        for i in range(n):
            ax = axes_right[i]
            if i < n - 1:
                ax.set_xlabel(None)
                ax.set_xticklabels([]) 
            else:
                ax.set_xlabel(xlabel)
    ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ### Save figure
    fig.savefig(options.figname, dpi=options.dpi, bbox_inches='tight')
    print(" Output", options.figname)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input parameters')

    parser.add_argument('--figname', dest='figname', type=str,
                        default="fig_scaling.png", help="figure name")
    
    parser.add_argument('--outfile', dest='outfile', type=str,
                        default="result_scaling.csv", help="output file name")
    
    parser.add_argument('--target', dest='target', type=str,
                        default="kspec", help="target")
    
    parser.add_argument('--dpi', dest='dpi', type=int,
                        default=600, help="dpi [600]")
    
    parser.add_argument('--min_epoch', dest='min_epoch', type=int,
                        default=100, help="min_epoch [100]")
    parser.add_argument('--min_ns', dest='min_ns', type=int,
                        default=0, help="min_ns [0]")
    
    parser.add_argument('--ylabel', dest='ylabel', type=str,
                        default=None, help="ylabel [None]")
    
    parser.add_argument('--ymin', dest='ymin', type=float,
                        default=None, help="ymin [None]")
    parser.add_argument('--ymax', dest='ymax', type=float,
                        default=None, help="ymax [None]")
    parser.add_argument('--yticks', dest='yticks', type=float,
                        default=None, help="yticks [None]")
    parser.add_argument('--myticks', dest='myticks', type=float,
                        default=None, help="myticks [None]")
    
    parser.add_argument('--mytick_positions', dest='mytick_positions', type=str,
                        default="3:5:15", help="positions of minor ticks [3:5:15]")

    args = parser.parse_args()

    main(args)
