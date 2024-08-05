import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import model.tool as tl
from model.utils import make_feature_names


def plot_result(model, data, setting, fsize=3.3, missing = None):
    dataset_name = setting['data_name']
    if setting['xticklabels'] is None:
        xticklabels = make_feature_names(model.k, model.dim_poly)
    else:  
        xticklabels = setting['xticklabels']
    
    if setting['yticklabels'] is None:
        yticklabels = xticklabels[:model.k]
    else:  
        yticklabels = setting['yticklabels']
    plt.rcParams['axes.xmargin'] = 0 
    annot = False
    fig_type = 'pdf'
    dir_path = f'./result/{model.fit_type}/{dataset_name}'
    os.makedirs(dir_path, exist_ok=True)
    
    if model.fit_type == 'Latent':
        w = np.concatenate(((model.A - np.eye(model.k)), model.F), axis=1)/setting['dt']
    else:
        
        w = np.concatenate((model.C @ (model.A - np.eye(model.k)), model.C @ model.F), axis=1)/setting['dt']
        
    if setting['gt'] is not None:
        gt_org = setting['gt']
        k_org = setting['gt'].shape[0]
        vmin = np.min(gt_org) - 0.1
        vmax = np.max(gt_org) + 0.1
        plt.rcParams["font.size"] = 12
        plt.rcParams['mathtext.fontset'] = 'cm'
        fig, (ax1, ax2) = plt.subplots(1, 2, 
                                       gridspec_kw=dict(width_ratios=[1,2.5], height_ratios=[1], wspace=0.1, hspace=0.3),
                                       figsize=(fsize-0.5, 0.7))
        sns.heatmap(gt_org[:,:k_org], 
                    xticklabels=xticklabels[:k_org],
                    yticklabels=yticklabels, vmin=vmin, vmax=vmax,
                    cmap='coolwarm', fmt ='1.1e', center=0.0, cbar=None, ax=ax1, annot=annot)
        ax1.tick_params(axis = 'x', labelrotation = 30)
        ax1.tick_params(axis = 'y', labelrotation = 0)
        ax1.tick_params(pad=0.5)
        
        sns.heatmap(gt_org[:,k_org:], 
                    xticklabels=xticklabels[k_org:],
                    yticklabels=[], vmin=vmin, vmax=vmax,
                    cmap='coolwarm', fmt ='1.1e', center=0.0, cbar=None, ax=ax2, annot=annot)
        ax2.tick_params(axis = 'x', labelrotation = 30)
        ax2.tick_params(axis = 'y', labelrotation = 0)
        ax2.tick_params(pad=0.5)
        # ax1.set_xlabel("RHS", fontsize=14)
        fig.savefig(f"./{dir_path}/ground_truth.{fig_type}", bbox_inches='tight', pad_inches=0.1)
        
        slabels = xticklabels
    else:
        vmin = np.min(w) - 0.1
        vmax = np.max(w) + 0.1
        slabels = make_feature_names(model.k, model.dim_poly)
    
    plt.rcParams["font.size"] = 12
    plt.rcParams['mathtext.fontset'] = 'cm'
    fig, (ax1, ax2) = plt.subplots(1, 2, 
                                    gridspec_kw=dict(width_ratios=[1,3], height_ratios=[1], wspace=0.1, hspace=0.3),
                                    figsize=(fsize, 0.7))
    sns.heatmap(w[:,:model.k], 
                xticklabels=slabels[:model.k],
                yticklabels=slabels[:model.k], vmin=vmin, vmax=vmax,
                cmap='coolwarm', fmt ='1.1e', center=0.0, cbar=None, ax=ax1, annot=annot)
    ax1.tick_params(axis = 'x', labelrotation = 30)
    ax1.tick_params(axis = 'y', labelrotation = 0)
    ax1.tick_params(pad=0.5)
    
    sns.heatmap(w[:,model.k:], 
                xticklabels=slabels[model.k:],
                yticklabels=[], vmin=vmin, vmax=vmax,
                cmap='coolwarm', fmt ='1.1e', center=0.0, ax=ax2, annot=annot)
    ax2.tick_params(axis = 'x', labelrotation = 30)
    ax2.tick_params(axis = 'y', labelrotation = 0)
    ax2.tick_params(pad=0.5)
    fig.savefig(f"./{dir_path}/st_weight_3_1.{fig_type}", bbox_inches='tight', pad_inches=0.1)
    
    
    if setting['gt'] is None:
        plt.rcParams["font.size"] = 12
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.figure(figsize=(0.7, fsize))
        sns.heatmap(model.C, 
                    xticklabels=xticklabels[:model.k], 
                    yticklabels=yticklabels, 
                    cmap='coolwarm', fmt ='1.1e', center=0.0, square=True)
        
        plt.xlabel("State")
        plt.savefig(f"./{dir_path}/group.{fig_type}", bbox_inches='tight', pad_inches=0.1)
        
    plt.rcParams["font.size"] = 28
    fig, ax = plt.subplots(figsize=(6.4,2.4))
    if missing is None:
        ax.plot(data)
    else:
        data_miss = data.copy()
        data_miss[missing] = np.nan
        data_no_miss = data.copy()
        data_no_miss[~missing] = np.nan
        ax.plot(data_no_miss)
        ax.plot(data_miss, linestyle='-', color='lightgray')
    
    ax.set_xlabel("Time", fontsize=28)
    ax.set_ylabel("Value", fontsize=28)
    fig.savefig(f"{dir_path}/org.{fig_type}", bbox_inches='tight', pad_inches=0.1)
        
    
    plt.rcParams["font.size"] = 28
    fig, ax = plt.subplots(figsize=(6.4,2.4))
    for i in range(model.k):
        ax.plot(model.Ez[:,i], zorder=2, label=f'$s_{i}$')
    ax.set_xlabel("Time", fontsize=28)
    ax.set_ylabel("Value", fontsize=28)
    fig.savefig(f"{dir_path}/smoothed_latent_dynamics.{fig_type}", bbox_inches='tight', pad_inches=0.1)
    
    
    with open(f'{dir_path}/model.pickle', mode='wb') as f:
        pickle.dump(model, f)
        

#--------------------------------#
def _plotResultsF(Snaps, outdir):
    Xorg=Snaps['Xorg']
    mn=np.nanmin(Xorg.flatten()); mx=np.nanmax(Xorg.flatten())
    tl.plt.clf()
    tl.plt.subplot(511)
    tl.plt.plot(Xorg,) # 'black')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Original')
    tl.plt.xticks([])
    tl.plt.subplot(512)
    tl.plt.plot(Xorg, 'lightgrey')
    tl.resetCol()
    tl.plt.plot(Snaps['Vf_full']) 
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Forecast')
    tl.plt.xticks([])
    tl.plt.subplot(513)
    tl.plt.plot(Snaps['Es_full'], 'lightgrey')
    tl.plt.plot(Snaps['Ef_full'], 'yellowgreen')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylabel('RMSE (cast)')
    tl.plt.xticks([])
    emx=np.nanmax(Snaps['Es_full'].flatten())
    tl.plt.ylim([0,emx])
    tl.plt.subplot(514)
    tl.plt.semilogy(Snaps['T_full'], '.', color='yellowgreen')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylabel('Speed')
    #tl.savefig("%sout_Vf"%(outdir),'pdf')
    tl.savefig("%sout_Vf"%(outdir),'png')
    tl.plt.close()

#--------------------------------#
def _plotResultsE(Snaps, outdir):
    Xorg=Snaps['Xorg']
    mn=np.nanmin(Xorg.flatten()); mx=np.nanmax(Xorg.flatten())
    tl.plt.clf()
    tl.plt.subplot(511)
    tl.plt.plot(Xorg) #, 'black')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Original')
    tl.plt.xticks([])
    tl.plt.subplot(512)
    tl.plt.plot(Xorg, 'lightgrey')
    tl.resetCol()
    tl.plt.plot(Snaps['Ve_full']) #, 'royalblue')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Est')
    tl.plt.xticks([])
    tl.plt.subplot(513)
    tl.plt.plot(Snaps['Ee_full'], 'yellowgreen')
    tl.plt.xlim([0,len(Xorg)])
    emx=np.nanmax(Snaps['Es_full'].flatten())
    tl.plt.ylim([0,emx])
    tl.plt.ylabel('RMSE (est)')
    tl.plt.xticks([])
    tl.plt.subplot(514)
    tl.plt.semilogy(Snaps['T_full'], '.', color='yellowgreen')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylabel('Speed')
    #tl.savefig("%sout_Ve"%(outdir),'pdf')
    tl.savefig("%sout_Ve"%(outdir),'png')
    tl.plt.close()
#--------------------------------#