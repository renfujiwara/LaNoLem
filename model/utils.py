import numpy as np
import itertools as it
from scipy.stats import norm
from numba import njit
import lmfit
#-----------------------------#
# lmfit (default)
XTL=1.e-8
FTL=1.e-8
MAXFEV=100 
#-----------------------------#

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


@njit('f8[:](f8[:],i8,i8[:,:])')
def make_state_vec(sta, k_nl, comb_list):
    vec = np.zeros(k_nl)
    if k_nl == 0: return vec
    tmp = np.append(1.0, sta)
    vec = np.array([np.prod(tmp[cmt]) for cmt in comb_list])
    return vec

# @njit('f8(f8[:,:],f8[:,:],f8[:,:,:])')
def log_multivariate_normal_density(data, mu_o, sgm):
    """Log probability for full covariance matrices. """
    # solve_triangular = linalg.solve_triangular
    min_covar=1.e-7
    llh = 0
    for (data_t, mu_t, sgm) in zip(data, mu_o, sgm):
        n_dim = len(data_t)
        # try:
        #     cv_chol = np.linalg.cholesky(cv, lower=True)
        # except linalg.LinAlgError:
        cv_chol = np.linalg.cholesky(sgm + min_covar * np.eye(n_dim))
        cv_log_det = 2 * np.sum(np.log(np.diag(cv_chol)))
        cv_sol = np.linalg.solve(cv_chol, (data_t - mu_t))
        llh += -0.5 * (np.sum(cv_sol ** 2) + n_dim * np.log(2 * np.pi) + cv_log_det)
        
    return llh

def compute_coding_cost(X, Y):
    diff = (X - Y).flatten().astype("float64")
    logprob = norm.logpdf(diff, loc=diff.mean(), scale=diff.std())
    cost = -1 * logprob.sum() / np.log(2.)
    return cost

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
    

def nl_fit(nlds, data, ftype): 
    nlds=_nl_fit(nlds, data, ftype) 
    return nlds 


def _nl_fit(nlds, data, ftype): 
    P=_createP(nlds, ftype)
    lmsol = lmfit.Minimizer(_distfunc, P, fcn_args=(data, nlds, ftype))
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


def _distfunc(P, data, nlds, ftype):
    n=data.shape[0]
    nlds=_updateP(P,nlds, ftype)
    (Sta, Obs)=nlds.gen(n) 
    diff=data.flatten() - Obs.flatten()
    diff[np.isnan(diff)]=0
    return diff