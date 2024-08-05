import os
import sys
import gc
import time
import math
from copy import deepcopy
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import numpy as np
from numpy.linalg import pinv
import itertools_len as itertools
from sklearn.metrics import root_mean_squared_error as rmse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from scipy.linalg import svd

from numba import njit
from tqdm import tqdm

from .utils import *
from .proximal_gradient import PG

h = 1.e-5
INF = 1.e+10


@njit(cache=True)
def _defunc(z, A, F, C, b, d, n, k_nl, comb_list):
    for t in range(0,n-1):
        z[t+1] = A @ z[t] + F @ make_state_vec(z[t], k_nl, comb_list) + b
    return z, z @ C.T + d


@njit(cache=True)
def _jacobian(z, A, F, j_delta, k_nl, comb_list):
    f_delta = j_delta + z
    b_delta = z - j_delta
    k = len(z)
    jacobian = np.zeros((k_nl, k))
    for i in range(k):
        jacobian[:,i] = (make_state_vec(f_delta[i], k_nl, comb_list) - make_state_vec(b_delta[i], k_nl, comb_list))
    return F @ jacobian / (2*h) + A


def update_mu0(Ez):
    return Ez[0].copy()


def update_Q0(Ez, Ezz):
    Q0 = Ezz[0] - np.outer(Ez[0], Ez[0])
    return Q0


def update_AF(init, Q, y, b, method, z, Szznl, Sz1znl, k, k_nl):
    return method.fit(init, y, b, z, Szznl, Sz1znl, Q, k, k_nl)


def update_C(Szz, Sxz, matrix_type):
    if(matrix_type == "diag"):
        return np.eye(Sxz.shape[0], Szz.shape[0])
    else:
        return np.dot(Sxz, pinv(Szz))


def update_b(y, z, znl, A, F):
    return np.mean(y - z @ A.T - znl @ F.T, axis=0)


def update_d(data, Ez, C):
    return np.mean(data - Ez @ C.T, axis=0)


@njit(cache=True)
def iter_Q(Ez, Ezz, Ez1z, missing, A, F, b, j_m, k_nl, comb_list):
    Q = np.zeros((len(A), len(A)))
    Ez1z_T = Ez1z.transpose(0,2,1)
    for t in range(len(Ez)-1):
        if missing[t] and missing[t+1]:
            A_nl = F @ j_m[t]
            ofset = F @ make_state_vec(Ez[t], k_nl, comb_list) + b - A_nl @ Ez[t]
            A_nl += A
            val = A_nl @ (Ez1z_T[t] - np.outer(Ez[t], ofset)) + np.outer(Ez[t+1], ofset)
            Q += A_nl @ Ezz[t] @ A_nl.T + np.outer(ofset, ofset) - val - val.T + Ezz[t+1]
    return Q
    

def update_Q(model, Ez, Ezz, Ez1z, missing):
    Q = iter_Q(Ez, Ezz, Ez1z, missing, model.A, model.F, model.b, model.j_m, model.k_nl, model.comb_list)
    return Q/(len(Ez) -1)


def update_R(data, Ez, Sxx, Szz, Sxz, C, d):
    val = C @ (Sxz.T - np.einsum('ij,l->jl',  Ez, d)) + np.einsum('ij,l->jl',  data, d)
    R = Sxx - val - val.T + C @ Szz @ C.T + np.outer(d, d)
    return R/len(data)


def fit_each(model_org, data, missing, max_iter, hyper_param, dim_list = None, return_model = False, multi = True):
    if dim_list is None:
        dim_list = model_org.dim_list
    loss_list = []
    Models = []
    Sxx = data[missing].T @ data[missing]
    Results = []
    if model_org.l1_r == 0:
        sparse = False
    else:
        sparse = True
    for dim_poly in dim_list:
        model = deepcopy(model_org)
        model.k = hyper_param[0]
        model.lam = hyper_param[1]
        model.trans_offset = hyper_param[2]
        model.k_nl, model.comb_list = get_k_nl(model.k, dim_poly)
        model.set_ks(dim_poly) 
        model.dim_poly = dim_poly
        model.init_params(data)
        model = model.initialize(data, missing)
        gd_method = PG(ptol = model.ptol, l1_lam = model.l1_lam, l2_lam = model.l2_lam, sparse = sparse)
        history_loss =[INF]
        
        model_d = deepcopy(model)
        loss_d = INF
        ascent = 0
        
        for iteration in range(max_iter):
            try:
                # inference-step
                model.forward(data, missing)
                model.backward()
                
                # learning-step
                model.solve(data, missing, Sxx, gd_method)
                loss = model.score(data, missing)
                model.loss = loss
            except (np.linalg.LinAlgError, FloatingPointError):
                conv = 'except'
                break
            
            # history_llh.append(llh)
            history_loss.append(loss)
            
            if loss < loss_d:
                model_d = deepcopy(model)
                loss_d = loss
            
            if abs(history_loss[-1] - history_loss[-2]) < model.tol:
                conv = 'conv1'
                break
            elif history_loss[-1] > history_loss[-2]:
                ascent += 1
                if ascent > model.MAX_ASCENT:
                    conv = 'conv2'
                    break
            else:
                ascent = 0
        else:
            conv = 'max'
        
        result = {"hyper_param": hyper_param}
        result["dim_poly"] = [dim_poly]
        param = model.A - np.eye(model.k)
        params = np.hstack((param, model.F))
        if ( (np.all(np.abs(model.F) < model.th))
            or (np.count_nonzero(np.abs(param) > model.th) < model.k)
            or (np.any(np.count_nonzero(np.abs(params) > model.th, axis=1) > 2*model.k))
            ):
            result["loss"] = INF
            result["mdl"] = INF
        else:
            result["loss"] = loss_d
            result["mdl"] = model_d.modeling_cost(data, missing)
        result["err"] = model_d.err(data)
        
        if model_org.lstep > 0:
            result["err_f"] = model_d.err_f(data)
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
            
        loss_list.append(result["loss"])
        Results.append(result)
        if return_model:
            model_d.history_loss = history_loss
            model_d.conv = conv
            model_d.iter = iteration
            model_d.mdl = result["mdl"]
            Models.append(model_d)
            
        
    best_i = np.argmin(loss_list)
    best_result = Results[best_i]
    if return_model:
        best_result["md"] = deepcopy(Models[best_i])
    
    if multi:
        del data
        del model
        del model_org
        del Sxx
        del gd_method
        del Models
        gc.collect()
        
    
    return best_result


@njit(cache=True)
def _filter(mu_t, data, missing, P, C, R, d, Ih, llh):
    sgm = C @ P @ C.T + R     
    inv_sgm = pinv(sgm)
    K = P @ C.T @ inv_sgm
    mu_o = C @ mu_t + d 
    dlt = data - mu_o
    if missing:
        mu = mu_t + K @ dlt
        sign, logdet = np.linalg.slogdet(inv_sgm)
        llh += sign * logdet * 0.5 - dlt @ inv_sgm @ dlt * 0.5 - 0.5 * len(d) * np.log(2 * np.pi)
    else:
        mu = mu_t.copy()
    K_hat = (Ih - K @ C)
    V = K_hat @ P @ K_hat.T + K @ R @ K.T    
    return mu, mu_o, sgm, V, llh


@njit(cache=True)
def _forward(data, missing, mu0, A, F, b, Q, Q0, C, R, d, j_delta, k, k_nl, comb_list):
    n, dim_data = data.shape
    Ih = np.eye(k)
    mu = np.empty((n, k))
    mu_t = np.zeros((n, k))
    mu_o = np.zeros((n, dim_data))
    sgm = np.empty((n, dim_data, dim_data))
    V = np.zeros((n, k, k))
    Phat = np.zeros((n, k, k))
    A_nl_mu = np.zeros((n,k,k))
    
    mu_t[0] = mu0.copy()
    llh = 0
    mu[0], mu_o[0], sgm[0], V[0], llh = _filter(mu0, data[0], missing[0], Q0, C, R, d, Ih, llh)
    
    for t in range(1, len(data)):
        mu_t[t] = A @ mu[t-1] + F @ make_state_vec(mu[t-1], k_nl, comb_list) + b
        A_nl_mu[t-1] = _jacobian(mu[t-1], A, F, j_delta, k_nl, comb_list)
        Phat[t] = A_nl_mu[t-1] @ V[t - 1] @ A_nl_mu[t-1].T + Q
        mu[t], mu_o[t], sgm[t], V[t], llh = _filter(mu_t[t], data[t], missing[t], Phat[t], C, R, d, Ih, llh)
        
    A_nl_mu[t] = _jacobian(mu[t], A, F, j_delta, k_nl, comb_list)
    return mu, mu_t, mu_o, sgm, V, Phat, A_nl_mu, llh



@njit(cache=True)
def _backward(mu, mu_t, V, Phat, A_nl, ks, k_d2, k_d3, k, k_nl, j_delta, comb_list):
    n = len(mu)
    Ez = np.zeros((n, k))
    Eznl = np.zeros((n, k_nl))
    Ezz = np.zeros((n, k, k))
    Ez1z = np.zeros((n, k, k))
    aug_z = np.zeros((n, k_nl+k))
    j_m = np.zeros((n, k_nl, k))
    
    P = V[-1].copy()
    Ez[-1] = mu[-1].copy()
    Ezz[-1] = P + np.outer(Ez[-1], Ez[-1])
    for t in range(n - 2, -1, -1):
        J = V[t] @ A_nl[t].T @ pinv(Phat[t+1])
        Ez[t] = mu[t] + J @ (Ez[t+1] - mu_t[t+1])
        Ez1z[t] = J @ P + np.outer(Ez[t + 1], Ez[t])
        P = V[t] + J @ (P - Phat[t+1]) @ J.T
        Ezz[t], Eznl[t] = moment(Ez[t], P, k, ks, k_d2, k_d3, k_nl)
        j_m[t] = diff_m(Ez[t], j_delta, k_nl, comb_list)
    aug_z = np.hstack((Ez, Eznl))
    
    return Ez, Eznl, aug_z, Ezz, Ez1z, j_m/(2*h)


class NLDS:
    def __init__(self, dim_list=[2, 3, 4], tol = 1.e-2, ptol = 1.e-10, th = 1.e-2, num_works=-1, print_log = False, verbose=False, 
                 random_state = 42):
        
        self.dim_list = dim_list
        self.tol = tol
        self.ptol = ptol
        self.th = th
        if num_works == -1:
            self.num_works = int(os.cpu_count()/4)
        else:
            self.num_works = num_works
        self.print_log = print_log
        self.verbose = verbose
        self.random_state = random_state
        self.conv = None
    
    
    def init_params(self, data):
        k = self.k
        dim_data = self.dim_data
        k_nl = self.k_nl
        rand = np.random.default_rng(self.random_state)
        if(self.obs_matrix == "diag"):
            self.C = np.eye(dim_data)
            self.mu0 = np.zeros(k) 
            self.A = np.eye(k) #+ rand.normal(0, 1, size=(k, k)) * 1e-7
            self.F = np.zeros((k,k_nl))
            self.b = np.zeros(k)
            self.d = np.zeros(dim_data)
            self.Q = np.eye(k)
            self.Q0 = np.eye(k) * 1e+7 
            self.R = np.eye(dim_data)
            self.MAX_ASCENT = 5 
        else:
            # u, sig, v = svd(data, full_matrices=False)
            # self.C = v[:k].T
            # data_s = u[:,:dim_data] @ np.diag(sig)
            # self.mu0 = data_s[0, :k]
            self.C = np.eye(dim_data, k) #+ rand.normal(0, 1, size=(dim_data, k))
            self.mu0 = np.zeros(k) 
            self.A = np.eye(k) #+ rand.normal(0, 1, size=(k, k))
            self.F = np.zeros((k,k_nl))
            self.b = np.zeros(k)
            self.d = np.zeros(dim_data)
            self.Q = np.eye(k) * 1.e-2
            self.Q0 = np.eye(k) * 1.e-2
            self.R = np.eye(dim_data) * 1.e-2
            self.MAX_ASCENT = 100
        
        self.history_loss = []
        self.history_llh = []
            
        
    def make_feature_names(self):
        return make_feature_names(self.k, self.dim_poly)
        
    
    def initialize(self, data, missing):
        self.n, self.dim_data = np.shape(data)
        k = self.k
        self.j_delta = np.diag(np.full(k, h, dtype=float))
        self.l1_lam = self.lam * self.l1_r
        self.l2_lam = self.lam * self.l2_r
        missing_r = np.roll(missing, 1)
        missing_l = np.roll(missing, -1)
        self.missing_z1 = np.logical_and(missing_r, missing)
        self.missing_z1[0] = False
        self.missing_z = np.logical_and(missing_l, missing)
        self.missing_z[-1] = False
        
        
        return self
    
    
    def jacobian(self, z):
        return _jacobian(z, self.A, self.F, self.j_delta, self.k_nl, self.comb_list)
        

    def forward(self, data, missing):
        self.mu, self.mu_t, self.mu_o, self.sgm, self.V, self.Phat, self.A_nl_mu, self.llh = _forward(data, missing, self.mu0, self.A, self.F, self.b, self.Q, self.Q0, 
                                                                           self.C, self.R, self.d, self.j_delta, self.k, self.k_nl, self.comb_list)
            
        
    def backward(self):
        self.Ez, self.Eznl, self.aug_z, self.Ezz, self.Ez1z, self.j_m = _backward(self.mu, self.mu_t, self.V, self.Phat, self.A_nl_mu, self.ks, self.k_d2, self.k_d3,
                                                                      self.k, self.k_nl, self.j_delta, self.comb_list)
                

    def solve(self, data, missing, Sxx, method):
        missing_z1 = self.missing_z1
        missing_z = self.missing_z
        
        Ez = self.Ez
        Ezz = self.Ezz
        Eznl = self.Eznl
        Sxz = data[missing].T @ self.Ez[missing]
        Szz = np.sum(Ezz[missing], axis=0)
        y = Ez[missing_z1] - self.b
        z = self.aug_z[missing_z]
        Szznl = z.T @ z
        Sz1znl = self.Ez[missing_z1].T @ z
        
        
        self.mu0 = Ez[0].copy() 
        self.Q0 = update_Q0(Ez, Ezz) 
        
        self.A, self.F = update_AF(np.hstack((self.A, self.F)), self.Q, y, self.b, 
                                   method, z, Szznl, Sz1znl, self.k, self.k_nl)
        
        self.C = update_C(Szz, Sxz, self.obs_matrix)
        
        if(self.trans_offset):
            self.b = update_b(Ez[missing_z1], Ez[missing_z], Eznl[missing_z], self.A, self.F)
        if(self.obs_offset):
            self.d = update_d(data[missing], Ez[missing], self.C)
        
        self.R = update_R(data[missing], Ez[missing], Sxx, Szz, Sxz, self.C, self.d)
        self.Q = update_Q(self, Ez, Ezz, self.Ez1z, missing)
    
    
    def print_params(self):
        print(f"mu0:{self.mu0}\n")
        print(f"A:{self.A}\n")
        print(f"F:{self.F}\n")
        print(f"Q:{self.Q}\n")
        print(f"C:{self.C}\n")
        print(f"d:{self.d}\n")
        print(f"R:{self.R}\n")
        
        
    def print_result(self, time=None):
        if self.print_log:
            print(f"loss: {self.loss}, mdl: {self.mdl}")
            print(f"lambda: {self.lam}")
            if time is not None:
                print(f'process time: {time}')
            print(f"dim: {self.dim_poly}, k: {self.k}")
            print(f'iter_num: {self.iter}')
        
        
    def set_ks(self, dim_poly):
        ks = np.ones(2) * self.k
        if dim_poly == 2:
            ks[0] = -1
            ks[1] = -1
        elif dim_poly ==3:
            ks[0] = self.k
            ks[1] = -1
        elif dim_poly == 4:
            ks[0] = self.k
            ks[1] = self.k
        self.ks = ks
        self.k_d2 = math.comb(self.k + 1, 2)
        self.k_d3 = math.comb(self.k + 2, 3)

    
    def modeling_cost(self, data, missing):
        I = np.eye(self.k, self.k)
        
        cost = compute_model_cost(self.A) + compute_model_cost(self.F) + compute_model_cost(self.C) 
        cost += compute_model_cost(self.Q) + compute_model_cost(self.R) + compute_model_cost(self.mu0) + compute_model_cost(self.Q0)
        if self.trans_offset:
            cost += compute_model_cost(self.b)
        if self.obs_offset:
            cost += compute_model_cost(self.d) 
        
        try:
            self.forward(data, missing)
            cost += compute_coding_cost(data[missing], self.mu_o[missing]) 
        except (np.linalg.LinAlgError, FloatingPointError):
            cost = INF
            
        if math.isnan(cost): cost = INF
        return cost
    
    
    def loglikelihood(self, data, missing):
        try:
            self.forward(data, missing)
        except:
            self.llh = -INF
        return self.llh

    
    def err(self, data):
        try:
            _, Obs = self.gen()
            if np.any(np.isnan(Obs)):
                return None
            else:
                return rmse(data, Obs)
        except:
            return None
        
        
        
    def err_f(self, data):
        try:
            self = nl_fit(self, data, "si")
            _, Obs = self.gen(len(data)+self.lstep)
            th=5.0
            mx=max(abs(data.flatten()))
            if(sum(abs(Obs.flatten())>mx*th)>=1):
                return None
            else:
                return rmse(data, Obs[:len(data)])
        except:
            return None
    
            
    def l1_l2(self):
        params = np.hstack((self.A - np.eye(self.k), self.F))
        return l1(self.l1_lam, params) + l2(self.l2_lam, params)
        
        
    def score(self, data, missing, llh=None):
        if llh is None:
            llh = self.loglikelihood(data, missing)
        return - llh + self.l1_l2() 
    
            
    def search_and_fit(self, data, missing, max_iter, hyper_params):
        num_works = np.minimum(self.num_works, len(hyper_params))
        if self.num_works > 1:
            if self.verbose:
                results = {}
                with tqdm(total=len(hyper_params)) as pbar:
                    with ProcessPoolExecutor(max_workers=num_works, mp_context=multiprocessing.get_context('spawn')) as executor:
                        futures = {executor.submit(fit_each, self, data, missing, max_iter, hyper_param): hyper_param for hyper_param in hyper_params}
                        for future in as_completed(futures):
                            arg = futures[future]
                            results[arg] = future.result()  
                            pbar.update(1)
            else:
                with ProcessPoolExecutor(max_workers=num_works, mp_context=multiprocessing.get_context('spawn')) as executor:
                        futures = {executor.submit(fit_each, self, data, missing, max_iter, hyper_param): hyper_param for hyper_param in hyper_params}
                        results = {}
                        for future in as_completed(futures):
                            arg = futures[future]
                            results[arg] = future.result()  

        else:
            results = {}
            if self.verbose:
                for hyper_param in tqdm(hyper_params):
                    results[hyper_param] = fit_each(self, data, missing, max_iter, hyper_param, multi=False)
            else:
                for hyper_param in hyper_params:
                    results[hyper_param] = fit_each(self, data, missing, max_iter, hyper_param, multi=False)
        
        _, best_result = min(results.items(), key=lambda x: x[1][self.ctype])  
        if self.print_log:   
            print(best_result)
        best_f = fit_each(self, data, missing, max_iter, best_result['hyper_param'], dim_list=best_result['dim_poly'], return_model=True, multi=False)
        model = best_f["md"]
        model.best_mdl = best_f['mdl']
        
        return model
            
            
    def fit(self, x, missing, lstep=-1, max_iter=50, k = None, fit_type=None, fit_init=False,
            lams = [0.0, 1.0, 5.0, 1e+1, 5e+1, 1e+2, 5e+2, 1e+3], l1_r = 1.0, l2_r=0.5):
        data = x
        np.seterr(over="raise")
        self.n, dim_data = data.shape
        self.dim_data = dim_data
        trans_offset = [True, False]
        self.fit_type = fit_type
        if(fit_type == 'Robust'): 
            self.obs_matrix = "diag"
            k_list = [dim_data]
            self.search_k = False
            self.obs_offset = False
            self.ctype = "mdl"
        elif(fit_type == 'Latent'):
            self.obs_matrix = "full"
            if k is None:
                k_list = np.arange(2, dim_data+1)
            else:
                k_list = [k]
            self.search_k = True
            self.obs_offset = True
            self.ctype = "mdl"
        else:
            print('You need fit_type')
            sys.exit()
        
        self.lstep = lstep
        self.fit_type = fit_type
        self.l2_r = l2_r
        self.l1_r = l1_r
        hyper_params = itertools.product(k_list, lams, trans_offset, repeat=1)
        tic1 = time.process_time()
        model =  self.search_and_fit(data, missing, max_iter, hyper_params)
        mu0_org = model.mu0.copy()
        if fit_init:
            try:
                model = nl_fit(model, data, "si")
            except:
                pass
        tic2 = time.process_time()
        err = model.err(data)
        if err is None:
            model.conv = 'inf'
            model.mu0 = mu0_org
        model.print_result(time=tic2 - tic1)
        return model
    
    
    def gen(self, n=None, mu0=None):
        k = self.k
        if(n is None): n = self.n
        if(mu0 is None): mu0 = self.mu0
        z = np.zeros((n, k))
        z[0] = mu0.copy()
        
        return _defunc(z, self.A, self.F, self.C, self.b, self.d, n, self.k_nl, self.comb_list) 

    