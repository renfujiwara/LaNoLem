import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools as it
from scipy.stats import norm
from numba import njit
from sklearn.preprocessing import PolynomialFeatures

INF = 1.e+10
h = 1.e-5

def get_k_nl(k, dim):
    if dim == 1: 
        return 0, np.zeros((1,1),dtype='i8')
    sta = np.arange(k+1)
    comb_list = np.array(list(it.combinations_with_replacement(sta,dim))[k+1:], dtype='i8')
    return len(comb_list), comb_list

def make_feature_names(k, dim, features=None):
    if features is None:
        features = [f's_{i}' for i in range(k)]
    poly = PolynomialFeatures(degree=dim, include_bias=False)
    poly.fit(np.zeros((1,k)))
    names = poly.get_feature_names_out(features) 
    return ['$' + s + '$' for s in names]


@njit(cache=True)
def make_state_vec(sta, k_nl, comb_list):
    if k_nl == 0: return np.zeros(k_nl)
    tmp = np.append(1.0, sta)
    return np.array([np.prod(tmp[cmt]) for cmt in comb_list])


def compute_coding_cost(X, Y):
    diff = (X - Y).flatten().astype("float32")
    logprob = norm.logpdf(diff, loc=diff.mean(), scale=diff.std())
    cost = -1 * logprob.sum() / np.log(2.)
    return cost


@njit(cache=True)
def compute_model_cost(X, sparse=True, n_bits=32, zero=1e-5):
    if sparse:
        X_abs = np.abs(X)
        X_nonzero = np.count_nonzero(X_abs > zero)
    else:
        X_nonzero = X.size
    
    cost = n_bits
    for i in range(X.ndim):
        cost += np.log(X.shape[i])
    return X_nonzero * cost + np.log(X.size)


@njit(cache=True)
def l1(lam, params):
    return np.sum(np.abs(lam *params))


@njit(cache=True)
def l2(lam, params):
    return np.sum(np.power(lam *params, 2)) 


@njit(cache=True)
def moment(Ez, P, k, ks, k_d2, k_d3, k_nl):
    Ezz = P + np.outer(Ez, Ez)
    Eznl = np.zeros((k_nl))
    d0 = 0
    d1 = 0
    d2 = 0
    k3 = ks[0]
    k4 = ks[1]
    
    for i in range(k):
        for j in range(i,k):
            Eznl[d0] = Ezz[i][j]
            d0 += 1
            for l in range(j,k3):
                Eznl[k_d2+d1] = Ez[i]*Ez[j]*Ez[l] + Ez[i]*P[j][l] + Ez[j]*P[i][l] + Ez[l]*P[i][j]
                d1 += 1
                for m in range(l,k4):
                    Eznl[k_d2+k_d3+d2] = Ez[i]*(Ez[j]*(Ez[l]*Ez[m]+P[l][m]) + Ez[l]*P[j][m] + Ez[m]*P[j][l]) \
                            + P[i][j] * (Ez[l]*Ez[m] + P[m][l]) + P[i][m] * (Ez[j]*Ez[l] + P[j][l]) \
                            + P[i][l] * (P[j][m] + Ez[j]*Ez[m])
                    d2 += 1
    return Ezz, Eznl


@njit(cache=True)
def diff_m(z, j_delta, k_nl, comb_list):
    f_delta = j_delta + z
    b_delta = z - j_delta
    k = len(z)
    jacobian = np.zeros((k_nl, k))
    for i in range(k):
        jacobian[:,i] = make_state_vec(f_delta[i], k_nl, comb_list) - make_state_vec(b_delta[i], k_nl, comb_list)
        
    return jacobian


@njit(cache=True)
def jacobian(z, A, F, j_delta, k_nl, comb_list):
    f_delta = j_delta + z
    b_delta = z - j_delta
    k = len(z)
    jacobian = np.zeros((k_nl, k))
    for i in range(k):
        jacobian[:,i] = (make_state_vec(f_delta[i], k_nl, comb_list) - make_state_vec(b_delta[i], k_nl, comb_list))
    return F @ jacobian / (2*h) + A


def make_result(model, hyper_param, dim_poly, data, loss):
    result = {"hyper_param": hyper_param}
    result["dim_poly"] = [dim_poly]
    param = model.A - np.eye(model.k)
    params = np.hstack((param, model.F))
    mx = np.max(np.abs(params))
    result["loss"] = loss
    
    if loss >= INF:
        result["mdl"] = INF
        result["mdl_s"] = INF
    elif mx <= model.th:
        result["mdl"] = INF
        result["mdl_s"] = INF          
    else:
        result["mdl"] = model.modeling_cost(data)
        result["mdl_s"] = result["mdl"]
            
    result["err"] = model.err(data)
    
    if (result["err"] is None):
        result["err"] = INF
        result["mdl"] = INF
    return result


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
    size = 20
    if setting['gt'] is not None:
        gt_org = setting['gt']
        k_org = setting['gt'].shape[0]
        vmin = np.min(gt_org) - 0.1
        vmax = np.max(gt_org) + 0.1
        plt.rcParams["font.size"] = 24
        plt.rcParams['mathtext.fontset'] = 'cm'
        fig, (ax1, ax2) = plt.subplots(1, 2, 
                                       gridspec_kw=dict(width_ratios=[1,2.5], height_ratios=[1], wspace=0.1, hspace=0.3),
                                       figsize=(fsize-0.5, 0.7))
        sns.heatmap(gt_org[:,:k_org], 
                    xticklabels=xticklabels[:k_org],
                    yticklabels=yticklabels, vmin=vmin, vmax=vmax,
                    cmap='coolwarm', fmt ='1.1e', center=0.0, cbar=None, ax=ax1, annot=annot)
        ax1.tick_params(axis = 'x', labelrotation = 30, labelsize=size)
        ax1.tick_params(axis = 'y', labelrotation = 0, labelsize=size)
        ax1.tick_params(pad=0.5)
        
        sns.heatmap(gt_org[:,k_org:], 
                    xticklabels=xticklabels[k_org:],
                    yticklabels=[], vmin=vmin, vmax=vmax,
                    cmap='coolwarm', fmt ='1.1e', center=0.0, cbar=None, ax=ax2, annot=annot)
        ax2.tick_params(axis = 'x', labelrotation = 30, labelsize=size)
        ax2.tick_params(axis = 'y', labelrotation = 0, labelsize=size)
        ax2.tick_params(pad=0.5)
        # ax1.set_xlabel("RHS", fontsize=14)
        fig.savefig(f"./{dir_path}/ground_truth.{fig_type}")
        
        slabels = xticklabels
    else:
        vmin = np.min(w) - 0.1
        vmax = np.max(w) + 0.1
        slabels = make_feature_names(model.k, model.dim_poly)
    
    plt.rcParams["font.size"] = 24
    plt.rcParams['mathtext.fontset'] = 'cm'
    fig, (ax1, ax2) = plt.subplots(1, 2, 
                                    gridspec_kw=dict(width_ratios=[1,3], height_ratios=[1], wspace=0.1, hspace=0.3),
                                    figsize=(fsize, 0.7))
    sns.heatmap(w[:,:model.k], 
                xticklabels=slabels[:model.k],
                yticklabels=slabels[:model.k], vmin=vmin, vmax=vmax,
                cmap='coolwarm', fmt ='1.1e', center=0.0, cbar=None, ax=ax1, annot=annot)
    ax1.tick_params(axis = 'x', labelrotation = 30, labelsize=size)
    ax1.tick_params(axis = 'y', labelrotation = 0, labelsize=size)
    ax1.tick_params(pad=0.5)
    
    sns.heatmap(w[:,model.k:], 
                xticklabels=slabels[model.k:],
                yticklabels=[], vmin=vmin, vmax=vmax,
                cmap='coolwarm', fmt ='1.1e', center=0.0, ax=ax2, annot=annot)
    ax2.tick_params(axis = 'x', labelrotation = 30, labelsize=size)
    ax2.tick_params(axis = 'y', labelrotation = 0, labelsize=size)
    ax2.tick_params(pad=0.5)
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)
    fig.savefig(f"./{dir_path}/st_weight_3_1.{fig_type}")
    
    
    if setting['gt'] is None:
        plt.rcParams["font.size"] = 16
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.figure(figsize=(0.7, fsize))
        sns.heatmap(model.C, 
                    xticklabels=xticklabels[:model.k], 
                    yticklabels=yticklabels, 
                    cmap='coolwarm', fmt ='1.1e', center=0.0, square=True)
        
        plt.xlabel("State")
        plt.savefig(f"./{dir_path}/group.{fig_type}", pad_inches=0.1)
        
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
    fig.savefig(f"{dir_path}/org.{fig_type}", pad_inches=0.1)
        
    
    plt.rcParams["font.size"] = 28
    fig, ax = plt.subplots(figsize=(6.4,2.4))
    for i in range(model.k):
        ax.plot(model.Ez[:,i], zorder=2, label=f'$s_{i}$')
    ax.set_xlabel("Time", fontsize=28)
    ax.set_ylabel("Value", fontsize=28)
    fig.legend(loc='center', bbox_to_anchor=(.5, 1.1), ncol=4, fontsize=32)
    fig.savefig(f"{dir_path}/smoothed_latent_dynamics.{fig_type}", pad_inches=0.1)
    
    
    with open(f'{dir_path}/model.pickle', mode='wb') as f:
        pickle.dump(model, f)
