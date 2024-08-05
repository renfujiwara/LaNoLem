import numpy as np
import itertools as it
from scipy.stats import norm
from copy import deepcopy as dcopy
from numba import njit
import model.tool as tl
import lmfit
import math
#-----------------------------#
# lmfit (default)
XTL=1.e-8
FTL=1.e-8
MAXFEV=100 
INF = 1.e+10
#-----------------------------#
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
    
    input_names = ['']
    input_names += features
    names = [f'${i}$' for i in features]
    sta = np.arange(k+1)
    comb_list = np.array(list(it.combinations_with_replacement(sta,dim))[k+1:], dtype='i8')
    for cmt in comb_list:
        txt = [input_names[j] for j in cmt]
        names.append(f"${''.join(txt)}$")
    return names


@njit(cache=True)
def make_state_vec(sta, k_nl, comb_list):
    if k_nl == 0: return np.zeros(k_nl)
    tmp = np.append(1.0, sta)
    return np.array([np.prod(tmp[cmt]) for cmt in comb_list])


def compute_coding_cost(X, Y):
    diff = (X - Y).flatten().astype("float64")
    logprob = norm.logpdf(diff, loc=diff.mean(), scale=diff.std())
    cost = -1 * logprob.sum() / np.log(2.)
    return cost


@njit(cache=True)
def compute_model_cost(X, sparse=True, n_bits=64, zero=1e-8):
    if sparse:
        X_abs = np.abs(X)
        X_nonzero = np.count_nonzero(X_abs > zero)
    else:
        X_nonzero = X.size
    
    cost = n_bits
    for i in range(X.ndim):
        cost += np.log(X.shape[i])
    return X_nonzero * cost + np.log(X.size)
    

def nl_fit(nlds, data, ftype, wtype='uniform', incremental=False): 
    nlds_org=dcopy(nlds)
    nlds=_nl_fit(nlds, data, ftype, wtype, incremental) 
    if(_distfunc_rmse(nlds_org, data, 'org')<_distfunc_rmse(nlds, data, 'fit')): nlds=nlds_org #notfin
    return nlds 


def _nl_fit(nlds, data, ftype, wtype, incremental): 
    P=_createP(nlds, ftype)
    lmsol = lmfit.Minimizer(_distfunc, P, fcn_args=(data, nlds, ftype, wtype))
    if incremental:
        res=lmsol.leastsq(xtol=0.1, ftol=0.1, max_nfev=20)
    else:
        res=lmsol.leastsq(xtol=XTL, ftol=FTL, max_nfev=MAXFEV)
    nlds=_updateP(res.params, nlds, ftype)
    return nlds 


def _createP(nlds, ftype):
    P = lmfit.Parameters()
    k=nlds.k
    V=True
    for i in range(0,k):
        P.add('mu0_%i'%(i), value=nlds.mu0[i], vary=V)
    return P


def _updateP(P, nlds, ftype):
    k=nlds.k
    for i in range(0,k):
        nlds.mu0[i]=P['mu0_%i'%(i)].value
    return nlds


def _distfunc(P, data, nlds, ftype, wtype):
    n=data.shape[0]
    nlds=_updateP(P, nlds, ftype)
    (Sta, Obs)=nlds.gen(n) 
    diff=data.flatten() - Obs.flatten()
    if np.isnan(diff).any():
        raise ValueError()
    diff=diff*func_Weight(len(diff), wtype)
    return diff

def _distfunc_rmse(nlds, data, ftype):
    try:
        (Sta, Obs)=nlds.gen(len(data)) 
        err = tl.RMSE(data, Obs)
    except:
        err = tl.INF
    # print(f'err({ftype}):{err}')
    return err


def func_Weight(n, wtype):
    if(wtype=='uniform'):
        # uniform
        return 1.0*np.ones(n)
    elif(wtype=='linear'):
        # linear 
        return 1.0*np.arange(0,n)
    elif(wtype=='linear_inv'):
        # linear  (inverse)
        return 1.0*np.arange(n,0,-1)
    elif(wtype=='exp'):
        # exponential
        T=n/2.0; ticks=np.arange(0,n)
        val=np.exp(ticks/T)
        return val


@njit(cache=True)
def l1(lam, params):
    return np.sum(np.abs(lam *params))


@njit(cache=True)
def l2(lam, params):
    return np.sum(np.power(lam *params, 2)) 


def make_result(model, hyper_param, dim_poly, data, loss):
    result = {"hyper_param": hyper_param}
    result["dim_poly"] = [dim_poly]
    param = model.A - np.eye(model.k)
    params = np.hstack((param, model.F))
    threshold = np.max(np.abs(params)) * 1e-3
    if(model.fit_type == 'Robust'): 
        if ((np.count_nonzero(np.abs(model.F) > threshold) == 0) 
            or (np.count_nonzero(np.abs(param) >= model.th) < model.k)):
            result["loss"] = INF
            result["mdl"] = INF
            result["mdl_s"] = INF            
                        
        # elif (np.any(np.count_nonzero(np.abs(params) >= model.th, axis=1) > 2*model.k)):
        #     result["loss"] = INF
        #     result["mdl"] = INF
        #     result["mdl_s"] = model.modeling_cost(data)
            
        else:
            for row in params:
                if np.count_nonzero(np.abs(row) >= model.th) >= (model.k * 2):
                    result["loss"] = INF
                    result["mdl"] = INF
                    result["mdl_s"] = model.modeling_cost(data)
                    break
            else:    
                result["loss"] = loss
                result["mdl"] = model.modeling_cost(data)
                result["mdl_s"] = result["mdl"]
    # elif np.any(np.count_nonzero(np.abs(model.F) >= threshold, axis=1) > math.ceil(model.F.shape[1]/2)):
    #     result["loss"] = loss
    #     result["mdl"] = INF
    else:
        if ((np.count_nonzero(np.abs(model.F) > threshold) == 0) 
            or (np.count_nonzero(np.abs(param) >= model.th) < model.k)):
            result["loss"] = INF
            result["mdl"] = INF
            result["mdl_s"] = INF  
        result["loss"] = loss
        result["mdl"] = model.modeling_cost(data)
        result["mdl_s"] = result["mdl"]
    result["err"] = model.err(data)
    
    if model.lstep > 0:
        result["err_f"] = model.err_f(data)
        if (result["err_f"] is None):
            result["loss"] = INF
            result["mdl"] =  INF
            result["err"] = INF
        elif (result["err"] is None):
            result["loss"] =  INF
            result["err"] = INF
    else:
        if (result["err"] is None):
            result["loss"] =  INF
            result["err"] = INF
        result["err_f"] = result["err"]
    
    return result


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

