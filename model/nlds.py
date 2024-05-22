import time
import sys
from copy import deepcopy

import os
import gc

import math
import numpy as np
from numpy.linalg import inv
import itertools_len as itertools
from sklearn.metrics import mean_squared_error as mse
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import multiprocessing

from .utils import make_state_vec, make_feature_names, get_k_nl, nl_fit, compute_coding_cost, compute_model_cost
from .proximal_gradient import PG

from numba import njit
from tqdm import tqdm

h = 1.e-5
INF = 1.e+10


@njit("UniTuple(f8[:,:],2)(f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:],f8[:],i8,i8,i8[:,:])")
def _defunc(z, A, F, C, b, d, n, k_nl, comb_list):
    for t in range(0,n-1):
        z[t+1] = A @ z[t] + F @ make_state_vec(z[t], k_nl, comb_list) + b
    return z, z @ C.T + d


@njit("f8[:,:](f8[:],f8[:,:],f8[:,:],f8[:,:],i8,i8[:,:])")
def _jacobian(z, A, F, j_delta, k_nl, comb_list):
    f_delta = j_delta + z
    b_delta = z - j_delta
    k = len(z)
    jacobian = np.zeros((k, k))
    for i in range(k):
        jacobian[:,i] = F @ (make_state_vec(f_delta[i], k_nl, comb_list) - make_state_vec(b_delta[i], k_nl, comb_list))
    
    return jacobian/ (2*h) + A


def update_mu0(Ez):
    return Ez[0].copy()


def update_Q0(Ez, Ezz, covariance_type = "diag"):
    Q0 = Ezz[0] - np.outer(Ez[0], Ez[0])
    if covariance_type == "diag":
        return np.diag(np.diag(Q0))
    else:
        return Q0


def update_AF(AF, method, Q, Ez, aug_z, Szznl, Sz1znl, k, k_nl):
    return method.fit(AF, Ez, aug_z, Szznl, Sz1znl, Q, k, k_nl)


def update_C(Ezz, Sxz, matrix_type):
    if(matrix_type == "diag"):
        return np.eye(Sxz.shape[0], Ezz.shape[1])
    else:
        return np.dot(Sxz, inv(np.sum(Ezz, axis=0)))


def update_b(Ez, Eznl, A, F):
    return np.mean(Ez[1:] - Ez[:-1] @ A.T - Eznl[:-1] @ F.T, axis=0)


def update_d(data, Ez, C):
    return np.mean(data - Ez @ C.T, axis=0)


@njit("f8[:,:](f8[:,:],f8[:,:],f8[:,:,:],f8[:,:,:],f8[:,:],f8[:,:],f8[:],f8[:,:],i8,i8[:,:])")
def iter_Q(Ez, Szz, Ezz, Ez1z, A, F, b, j_delta, k_nl, comb_list):
    Q = Szz - Ezz[0]
    Ez1z_T = Ez1z.transpose(0,2,1)
    for t in range(len(Ez)-1):
        A_nl = _jacobian(Ez[t], A, F, j_delta, k_nl, comb_list)
        ofset = (A - A_nl) @ Ez[t] + F @ make_state_vec(Ez[t], k_nl, comb_list) + b
        val = A_nl @ Ez1z_T[t]
        val -= A_nl @ np.outer(Ez[t], ofset)
        val += np.outer(Ez[t+1], ofset)
        Q += - val - val.T + A_nl @ Ezz[t] @ A_nl.T + np.outer(ofset, ofset)
    
    return Q
    

def update_Q(model, Ez, Szz, Ezz, Ez1z, covariance_type):
    Q = iter_Q(Ez, Szz, Ezz, Ez1z, model.A, model.F, model.b, model.j_delta, model.k_nl, model.comb_list)
    
    if covariance_type == "diag":
        return np.diag(np.diag(Q))/(len(Ez) -1)
    else:
        return Q/(len(Ez)-1)


def update_R(data, Ez, Sxx, Szz, Sxz, C, d, covariance_type):
    val = C @ Sxz.T - C @ np.einsum('ij,l->jl',  Ez, d) + np.einsum('ij,l->jl',  data, d)
    
    R = Sxx - val - val.T + C @ Szz @ C.T + np.outer(d, d)
    if covariance_type == "diag":
        return np.diag(np.diag(R))/len(data)
    else:
        return R/len(data)


def fit_each(model_org, data, max_iter, hyper_param, return_model = False, multi = True):
    model = deepcopy(model_org)
    lams = hyper_param[0]
    dim_poly = hyper_param[1]
    model.k = hyper_param[2]
    model.init_cov = hyper_param[3]
    model.init_type = hyper_param[4]
    
    model.k_nl, model.comb_list = get_k_nl(model.k, dim_poly)
    model.dim_poly = dim_poly
    
    model.set_lam(lams)
    result = {"md":None, "err":None, "mdl":INF, "loss":None, "hyper_param": hyper_param}
    
    model.init_params(data)
    model, loss_init, llh = model.initialize(data)
    model.iter = 0
    history_loss =[loss_init]
    history_llh = [llh]
    conv = 'max'
    Sxx = data.T @ data
    gd_method = PG(ptol = model.ptol, lams = model.lams)
    
    best_model = deepcopy(model)
    best_loss = loss_init
    for iteration in range(max_iter):
        model.iter = iteration
        try:
            # inference-step
            model.forward(data)
            model.backward()
            
            # learning-step
            model.solve(data, Sxx, gd_method)

            loss = model.score(data)
            model.loss = loss
            llh = model.llh
        except (np.linalg.LinAlgError, FloatingPointError):
            conv = 'except'
            break
        
        history_llh.append(llh)
        history_loss.append(loss)
        
        if abs(history_loss[-1] - history_loss[-2]) < model.tol:
            conv = 'conv1'
            break
        elif history_loss[-1] - loss_init > model.tol * 1e+2:
            conv = 'conv2'
            break
        
        if loss < best_loss:
            best_model = deepcopy(model)
            best_loss = loss


    best_model.history_loss = history_loss
    best_model.history_llh = history_llh
    best_model.conv = conv
    
    if return_model:
        result["md"] = best_model
        
    loss = best_model.loss
    result["err"] = best_model.err(data)
    result["loss"] = loss
    best_model.mdl = best_model.modeling_cost(data, best_model.dim_poly)
    result["mdl"] = best_model.mdl
    
    if multi:
        del data
        del model
        del best_model
        del model_org
        del Sxx
        del history_llh
        del history_loss
        del gd_method
        gc.collect()
    
    return result


@njit("Tuple((f8[:],f8[:],f8[:,:],f8[:,:],f8))(f8[:],f8[:],f8[:,:],f8[:,:],f8[:,:],f8[:],f8[:,:])")
def _filter(mu_t, data, P, C, R, d, Ih):
    sgm = C @ P @ C.T + R     
    inv_sgm = inv(sgm)
    K = P @ C.T @ inv_sgm
    mu_o = C @ mu_t + d 
    dlt = data - mu_o
    
    mu = mu_t + K @ dlt
    K_hat = (Ih - K @ C)
    V = K_hat @ P @ K_hat.T + K @ R @ K.T
    df = dlt @ inv_sgm @ dlt * 0.5
    sign, logdet = np.linalg.slogdet(inv_sgm)
    llh = -0.5 * len(d) * np.log(2 * np.pi)
    llh += sign * logdet * 0.5 - df
    
    return mu, mu_o, sgm, V, llh


@njit("Tuple((f8[:,:],f8[:]))(f8[:],f8[:,:],i8,i8,i8)")
def moment(Ez, Vhat, k, k_nl, dim_poly):
    Ezz = Vhat + np.outer(Ez, Ez)
    Eznl = np.zeros((k_nl))
    d = 0
    if dim_poly >= 2:
        for i in range(k):
            for j in range(i,k):
                Eznl[d] = Ezz[i][j]
                d += 1
    
    if dim_poly >= 3:
        for i in range(k):
            for j in range(i,k):
                for l in range(j,k):
                    Eznl[d] = Ez[i]*Ez[j]*Ez[l] + Ez[i]*Vhat[j][l] + Ez[j]*Vhat[i][l] + Ez[l]*Vhat[i][j]
                    d += 1
                    
    if dim_poly >= 4:
        for i in range(k):
            for j in range(i,k):
                for l in range(j,k):
                    for m in range(l,k):
                        Eznl[d] = Ez[i]*Ez[j]*(Ez[l]*Ez[m]+Vhat[l][m]) + Ez[i]*Ez[l]*Vhat[j][m] \
                            + Ez[i]*Ez[m]*Vhat[j][l] + Ez[l]*Ez[m]*Vhat[i][j] \
                            + Vhat[i][j] * Vhat[m][l] + Vhat[i][m] * (Ez[j]*Ez[l] + Vhat[j][l]) + Vhat[i][l] * (Vhat[j][m] + Ez[j]*Ez[m])
                        d += 1
    return Ezz, Eznl


@njit("Tuple((f8[:,:],f8[:,:],f8[:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8))(f8[:,:],f8[:],f8[:,:],f8[:,:],f8[:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:],f8[:,:],i8,i8,i8[:,:])")
def _forward(data, mu0, A, F, b, Q, Q0, C, R, d, j_delta, k, k_nl, comb_list):
    n, dim_data = data.shape
    Ih = np.eye(k)
    mu = np.empty((n, k))
    mu_t = np.zeros((n, k))
    mu_o = np.empty((n, dim_data))
    sgm = np.empty((n, dim_data, dim_data))
    V = np.empty((n, k, k))
    P = np.zeros((n, k, k))
    A_nl_mu = np.zeros((n,k,k))
    
    mu_t[0] = mu0.copy()
    mu[0], mu_o[0], sgm[0], V[0], llh = _filter(mu0, data[0], Q0, C, R, d, Ih)
    
    for t in range(1, len(data)):
        mu_t[t] = A @ mu[t-1] + F @ make_state_vec(mu[t-1], k_nl, comb_list) + b
        A_nl_mu[t-1] = _jacobian(mu[t-1], A, F, j_delta, k_nl, comb_list)
        P[t-1] = A_nl_mu[t-1] @ V[t - 1] @ A_nl_mu[t-1].T + Q
        mu[t], mu_o[t], sgm[t], V[t], llh_t = _filter(mu_t[t], data[t], P[t-1], C, R, d, Ih)
        llh += llh_t
        
    return mu, mu_t, mu_o, sgm, V, P, A_nl_mu, llh

@njit("Tuple((f8[:,:],f8[:,:],f8[:,:],f8[:,:,:],f8[:,:,:]))(f8[:,:],f8[:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],i8,i8,i8)")
def _backward(mu, mu_t, V, P, A_nl, k, k_nl, dim_poly):
    n = len(mu)
    Ez = np.zeros((n, k))
    Eznl = np.zeros((n, k_nl))
    Ezz = np.zeros((n, k, k))
    Ez1z = np.zeros((n, k, k))
    aug_z = np.zeros((n, k_nl+k))
    
    Vhat = V[-1].copy()
    Ez[-1] = mu[-1].copy()
    Ezz[-1] = Vhat + np.outer(Ez[-1], Ez[-1])
    for t in range(n - 2, -1, -1):
        J = V[t] @ A_nl[t].T @ inv(P[t])
        Ez[t] = mu[t] + J @ (Ez[t+1] - mu_t[t+1])
        Ez1z[t] = Vhat @ J.T + np.outer(Ez[t + 1], Ez[t])
        Vhat = V[t] + J @ (Vhat - P[t]) @ J.T
        Ezz[t], Eznl[t] = moment(Ez[t], Vhat, k, k_nl, dim_poly)
    aug_z = np.hstack((Ez, Eznl))
    
    return Ez, Eznl, aug_z, Ezz, Ez1z #zznl, z1znl

def model_compress(model, dim_poly):
    model.k_nl, model.comb_list = get_k_nl(model.k, dim_poly)
    model.F = model.F[:, :model.k_nl]
    return model


class NLDS:
    def __init__(self, dim_list=[2, 3, 4], lam_list=[1e-3, 1e-2, 1e-1, 1.0, 1e+1, 1e+2], dt=1.0, init_cov_list = [1e+1, 1.0, 1e-1, 1e-2, 1e-3],
                 tol = 1.0, ptol = 1.e-20, init_state_cov = "full", trans_cov = "full", obs_cov = "full",
                 num_works=-1, print_log = False, verbose=False, random_state = 42, naive=True):
        
        self.dim_list = dim_list
        self.comb_lam = lam_list
        self.dt = dt
        self.init_cov_list = init_cov_list
        self.tol = tol
        self.ptol = ptol
        self.init_state_cov = init_state_cov
        self.trans_cov = trans_cov
        self.obs_cov = obs_cov
        if num_works == -1:
            self.num_works = int(os.cpu_count()/2)
        else:
            self.num_works = num_works
        self.print_log = print_log
        self.verbose = verbose
        self.random_state = random_state
        self.naive = naive
    
    def init_params(self, data):
        k = self.k
        dim_data = self.dim_data
        k_nl = self.k_nl
        
        if(self.obs_matrix == "diag"):
            self.C = np.eye(dim_data)
            if self.init_type:
                self.mu0 = data[0].copy()
            else:
                self.mu0 = np.zeros(k)
            self.A = np.eye(k)
            self.F = np.zeros((k,k_nl))
        else:
            if self.naive:
                rand = np.random.default_rng(self.random_state)
                self.C = np.eye(dim_data, k) + rand.normal(0, 1, size=(dim_data, k))
                self.mu0 = rand.normal(0, 1 ,size = k)
            else:
                u, sig, v = np.linalg.svd(data, full_matrices=False)
                self.C = v[:k].T
                data_s = u[:,:dim_data] @ np.diag(sig)
                self.mu0 = data_s[0, :k]
                
            self.A = np.eye(k)
            self.F = np.zeros((k,k_nl))
            
        self.Q0 = np.eye(k) * self.init_cov
        self.Q = np.eye(k) * self.init_cov
        self.R = np.eye(dim_data) * self.init_cov
        self.b = np.zeros(k)
        self.d = np.zeros(dim_data)
            
        self.history_loss = []
        self.history_llh = []
            
        
    def make_feature_names(self):
        return make_feature_names(self.k, self.dim_poly)
    
    
    def set_lam(self, lams):
        self.l_lam = lams
        self.nl_lam = lams
        l_lam = np.ones((self.k,self.k)) * lams
        nl_lam = np.ones((self.k,self.k_nl)) * lams
        self.lams = np.hstack((l_lam, nl_lam))
        
    
    def initialize(self, data):
        self.n, self.dim_data = np.shape(data)
        k = self.k
        self.j_delta = np.diag(np.full(k, h, dtype=float))

        loss = self.score(data)
        self.loss = loss
        llh = self.llh

        return self, loss, llh
    
    
    def jacobian(self, z):
        return _jacobian(z, self.A, self.F, self.j_delta, self.k_nl, self.comb_list)
        

    def forward(self, data):
        self.mu, self.mu_t, self.mu_o, self.sgm, self.V, self.P, self.A_nl_mu, self.llh = _forward(data, self.mu0, self.A, self.F, self.b, self.Q, self.Q0, 
                                                                           self.C, self.R, self.d, self.j_delta, self.k, self.k_nl, self.comb_list)
            
        
    def backward(self):
        self.Ez, self.Eznl, self.aug_z, self.Ezz, self.Ez1z = _backward(self.mu, self.mu_t, self.V, self.P, self.A_nl_mu, 
                                                                      self.k, self.k_nl, self.dim_poly)
        
        self.Szznl = np.einsum('ij,il->jl', self.aug_z[:-1], self.aug_z[:-1])
        self.Sz1znl = np.einsum('ij,il->jl', self.Ez[1:], self.aug_z[:-1])


    def solve(self, data, Sxx, method):
        Ez = self.Ez
        Ezz = self.Ezz
        Eznl = self.Eznl
        Sxz = np.einsum('ij,il->jl', data, self.Ez)
        Szz = np.sum(Ezz, axis=0)
        Szznl = self.Szznl
        Sz1znl = self.Sz1znl

        # Initial zte mean/covariance
        self.mu0 = Ez[0].copy() 
        self.Q0 = update_Q0(Ez, Ezz, self.init_state_cov)
        
        self.A, self.F = update_AF(np.hstack((self.A, self.F)), method, self.Q, 
                                   self.Ez, self.aug_z, Szznl, Sz1znl, self.k, self.k_nl)
        
        self.C = update_C(Ezz, Sxz, self.obs_matrix)
        
        self.b = update_b(Ez, Eznl, self.A, self.F)
        if(self.obs_offset):
            self.d = update_d(data, Ez, self.C)
        
        self.R = update_R(data, Ez, Sxx, Szz, Sxz, self.C, self.d, self.obs_cov)
        self.Q = update_Q(self, Ez, Szz, Ezz, self.Ez1z, self.trans_cov)
    
    
    def print_params(self):
        print(f"mu0:{self.mu0}\n")
        print(f"A:{self.A}\n")
        print(f"F:{self.F}\n")
        print(f"Q:{self.Q}\n")
        print(f"C:{self.C}\n")
        print(f"d:{self.d}\n")
        print(f"R:{self.R}\n")
        
    def print_result(self):
        print(f"loss: {self.loss}, mdl: {self.mdl}")
        print(f"lambda(lin): {self.l_lam}, lambda(nlin): {self.nl_lam}")
        print(f'process time: {self.time}')
        
    
    def modeling_cost(self, data, dim_poly):
        model = deepcopy(self)
        model = model_compress(model, dim_poly)

        cost = compute_model_cost(model.A) + compute_model_cost(model.F) + compute_model_cost(model.C) + compute_model_cost(model.b)
        if model.obs_offset:
            cost += compute_model_cost(model.d) 

        try:
            _, Obs = model.gen()
            cost += compute_coding_cost(data, Obs) 
        except FloatingPointError:
            cost = INF
            
        if math.isnan(cost): cost = INF
        return cost
    
    
    def loglikelihood(self, data):
        self.forward(data)
        return self.llh
    
    
    def err(self, data):
        try:
            _, Obs = self.gen()
            return mse(data, Obs) + self.l1()
        except:
            return INF
        
    def compress(self, data):
        best_mdl = self.modeling_cost(data, self.dim_poly)
        best_dim = self.dim_poly
        for dim_poly in range(self.dim_poly-1, 1, -1):
            mdl = self.modeling_cost(data, dim_poly)
            if best_mdl >= mdl:
                best_mdl = mdl
                best_dim = dim_poly
        model = model_compress(self, best_dim)
        model.mdl = best_mdl 
        model.dim_poly = best_dim
        return model
    
            
    def l1(self):
        return np.sum(self.lams * np.abs(np.hstack((self.A-np.eye(self.k),self.F))))
        
        
    def score(self, data, llh=None):
        if llh is None:
            llh = self.loglikelihood(data)
        return - llh + self.l1()
    
            
    def search_and_fit(self, data, max_iter, hyper_params):
        best_f = None
        if self.num_works > 1:
            if self.verbose:
                results = {}
                with tqdm(total=len(hyper_params)) as pbar:
                    with ProcessPoolExecutor(max_workers=self.num_works, mp_context=multiprocessing.get_context('spawn')) as executor:
                        futures = {executor.submit(fit_each, self, data, int(max_iter/2), hyper_param, False): hyper_param for hyper_param in hyper_params}
                        for future in concurrent.futures.as_completed(futures):
                            arg = futures[future]
                            results[arg] = future.result()  
                            pbar.update(1)
            else:
                with ProcessPoolExecutor(max_workers=self.num_works, mp_context=multiprocessing.get_context('spawn')) as executor:
                        futures = {executor.submit(fit_each, self, data, int(max_iter/2), hyper_param, False): hyper_param for hyper_param in hyper_params}
                        results = {}
                        for future in concurrent.futures.as_completed(futures):
                            arg = futures[future]
                            results[arg] = future.result()  

        else:
            results = {}
            if self.verbose:
                for hyper_param in tqdm(hyper_params):
                    results[hyper_param] = fit_each(self, data, int(max_iter/2), hyper_param, return_model=False, multi=False)
            else:
                for hyper_param in hyper_params:
                        results[hyper_param] = fit_each(self, data, int(max_iter/2), hyper_param, return_model=False, multi=False)
        
        i, best_f = min(results.items(), key=lambda x: x[1]['mdl'])     
        
        best_f = fit_each(self, data, max_iter, best_f['hyper_param'], return_model=True, multi=False)
        model = best_f["md"]
        model.best_mdl = best_f['mdl']
        
        return model
            
            
    def fit(self, x, y=None, max_iter=30, k = None, fit_type=None, fit_init=True, batch=False):
        data = x
        np.seterr(over="raise")
        self.n, dim_data = data.shape
        self.dim_data = dim_data
        if(fit_type == 'Robust'): 
            self.obs_matrix = "diag"
            k_list = [dim_data]
            self.search_k = False
            self.obs_offset = False
        elif(fit_type == 'Latent'):
            self.obs_matrix = "full"
            if k is None:
                k_list = np.arange(2, dim_data+1)
            else:
                k_list = [k]
            self.search_k = True
            self.obs_offset = True
        else:
            print('You need fit_type')
            sys.exit()
        
        self.fit_type = fit_type
        hyper_params = itertools.product(self.comb_lam, self.dim_list, k_list, self.init_cov_list, [True, False], repeat=1)
        tic1 = time.process_time()
        model =  self.search_and_fit(data, max_iter, hyper_params)
        
        model = model.compress(data)
        try:
            if fit_init:
                model = nl_fit(model, data, "si")
        except:
            pass
        tic2 = time.process_time()
        model.time = tic2 - tic1
        model.print_result()
        return model
    
    
    def gen(self, n=None, mu0=None):
        k = self.k
        if(n is None): n = self.n
        if(mu0 is None): mu0 = self.mu0
        z = np.zeros((n, k))
        z[0] = mu0.copy()
        
        return _defunc(z, self.A, self.F, self.C, self.b, self.d, n, self.k_nl, self.comb_list) 

    