from copy import deepcopy

import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error as mse

from numba import njit
from .utils import nl_fit

INF = 1.e+10


@njit(cache=True)
def _defunc(z, A, C, b, d, n):
    for t in range(0,n-1):
        z[t+1] = A @ z[t] + b
    return z, z @ C.T + d


def update_mu0(Ez):
    return Ez[0].copy()


def update_Q0(Ez, Ezz, covariance_type = "diag"):
    Q0 = Ezz[0] - np.outer(Ez[0], Ez[0])
    if covariance_type == "diag":
        return np.diag(np.diag(Q0))
    else:
        return Q0


def update_A(Ez1z, Ezz):
    return np.dot(np.sum(Ez1z, axis=0), inv(np.sum(Ezz, axis=0)))


def update_C(Szz, Sxz):
    return np.dot(Sxz, inv(Szz))


def update_b(y, x, A):
    return np.mean(y - x @ A.T, axis=0)


def update_d(y, x, C):
    return np.mean(y - x @ C.T, axis=0)


def update_Q(Ez, Ez1, Szz, Ezz, Ez1z, A, b, covariance_type):
    Sz1z = np.sum(Ez1z, axis=0)
    val = A @ (Sz1z.T - np.einsum('ij,l->jl',  Ez, b)) + np.einsum('ij,l->jl',  Ez1, b)
    Q = Szz - Ezz[0] - val - val.T + A @ Sz1z @ A.T + np.outer(b, b)
    if covariance_type == "diag":
        return np.diag(np.diag(Q))/(len(Ez)-1)
    else:
        return Q/(len(Ez)-1)


def update_R(data, Ez, Sxx, Szz, Sxz, C, d, covariance_type):
    val = C @ (Sxz.T - np.einsum('ij,l->jl',  Ez, d)) + np.einsum('ij,l->jl',  data, d)
    R = Sxx - val - val.T + C @ Szz @ C.T + np.outer(d, d)
    if covariance_type == "diag":
        return np.diag(np.diag(R))/len(data)
    else:
        return R/len(data)


def fit_each(model, data, missing, max_iter):
    model = deepcopy(model)
    model.init_params(data)
    model, llh = model.initialize(data, missing)
    model.iter = 0
    history_llh = [llh]
    conv = 'max'
    Sxx = data[missing].T @ data[missing]
    
    for iteration in range(max_iter):
        model.iter = iteration
        try:
        # inference-step
            model.forward(data, missing)
            model.backward()
            # learning-step
            model.solve(data, missing, Sxx)

            loss = model.score(data)
            model.loss = loss
            llh = model.llh
        except (np.linalg.LinAlgError, FloatingPointError):
            conv = 'except'
            break
        
        history_llh.append(llh)
        
        if history_llh[-1] > history_llh[-2]:
            conv = 'conv1'
            break
            
    model.conv = conv
    return model


@njit(cache=True)
def _filter(mu_t, data, missing, P, C, R, d, Ih):
    sgm = C @ P @ C.T + R     
    inv_sgm = inv(sgm)
    K = P @ C.T @ inv_sgm
    mu_o = C @ mu_t + d 
    if missing:
        dlt = data - mu_o
        mu = mu_t + K @ dlt
        sign, logdet = np.linalg.slogdet(inv_sgm)
        llh = sign * logdet * 0.5 - dlt @ inv_sgm @ dlt * 0.5 - 0.5 * len(d) * np.log(2 * np.pi)
    else:
        mu = mu_t.copy()
        llh = 0
    K_hat = (Ih - K @ C)
    V = K_hat @ P @ K_hat.T + K @ R @ K.T
    
    return mu, mu_o, sgm, V, llh


@njit(cache=True)
def _forward(data, missing, mu0, A, b, Q, Q0, C, R, d, k):
    n, dim_data = data.shape
    Ih = np.eye(k)
    mu = np.empty((n, k))
    mu_t = np.zeros((n, k))
    mu_o = np.empty((n, dim_data))
    sgm = np.empty((n, dim_data, dim_data))
    V = np.empty((n, k, k))
    P = np.zeros((n, k, k))
    
    mu_t[0] = mu0.copy()
    mu[0], mu_o[0], sgm[0], V[0], llh = _filter(mu0, data[0], missing[0], Q0, C, R, d, Ih)
    
    for t in range(1, len(data)):
        mu_t[t] = A @ mu[t-1] + b
        P[t-1] = A @ V[t - 1] @ A.T + Q
        mu[t], mu_o[t], sgm[t], V[t], llh_t = _filter(mu_t[t], data[t], missing[t], P[t-1], C, R, d, Ih)
        llh += llh_t
        
    return mu, mu_t, mu_o, sgm, V, P, llh


@njit(cache=True)
def _backward(mu, mu_t, V, P, A, k):
    n = len(mu)
    Ez = np.zeros((n, k))
    Ezz = np.zeros((n, k, k))
    Ez1z = np.zeros((n, k, k))
    
    Vhat = V[-1].copy()
    Ez[-1] = mu[-1].copy()
    Ezz[-1] = Vhat + np.outer(Ez[-1], Ez[-1])
    for t in range(n - 2, -1, -1):
        J = V[t] @ A.T @ inv(P[t])
        Ez[t] = mu[t] + J @ (Ez[t+1] - mu_t[t+1])
        Ez1z[t] = J @ Vhat + np.outer(Ez[t + 1], Ez[t])
        Vhat = V[t] + J @ (Vhat - P[t]) @ J.T
        Ezz[t] = Vhat + np.outer(Ez[t], Ez[t])
    
    return Ez, Ezz, Ez1z


class LDS:
    def __init__(self, init_state_cov = "full", trans_cov = "full", obs_cov = "full",
                 print_log = False, verbose=False, random_state = 42):
        
        self.init_state_cov = init_state_cov
        self.trans_cov = trans_cov
        self.obs_cov = obs_cov
        self.print_log = print_log
        self.verbose = verbose
        self.random_state = random_state
    
    def init_params(self, data):
        k = self.k
        dim_data = self.dim_data
        # rand = np.random.default_rng(self.random_state)
        self.C = np.eye(dim_data, k) #+ rand.normal(0, 1, size=(dim_data, k))
        self.mu0 = np.zeros(k)
            # else:
            #     u, sig, v = np.linalg.svd(data, full_matrices=False)
            #     self.C = v[:k].T
            #     data_s = u[:,:dim_data] @ np.diag(sig)
            #     self.mu0 = data_s[0, :k]
                
        self.A = np.eye(k)
        self.Q0 = np.eye(k)
        self.Q = np.eye(k)
        self.R = np.eye(dim_data)
        self.b = np.zeros(k)
        self.d = np.zeros(dim_data)
            
        self.history_llh = []
        
    
    def initialize(self, data, missing):
        self.n, self.dim_data = np.shape(data)
        llh = self.score(data)
        self.llh = llh
        missing_r = np.roll(missing, 1)
        missing_l = np.roll(missing, -1)
        self.missing_z1 = np.logical_and(missing_r, missing)
        self.missing_z1[0] = False
        self.missing_z = np.logical_and(missing_l, missing)
        self.missing_z[-1] = False

        return self, llh
    
    

    def forward(self, data, missing):
        self.mu, self.mu_t, self.mu_o, self.sgm, self.V, self.P, self.llh = _forward(data, missing, self.mu0, self.A, self.b, self.Q, 
                                                                                     self.Q0, self.C, self.R, self.d, self.k)
            
        
    def backward(self):
        self.Ez, self.Ezz, self.Ez1z = _backward(self.mu, self.mu_t, self.V, self.P, self.A, self.k)
        


    def solve(self, data, missing, Sxx):
        missing_z1 = self.missing_z1
        missing_z = self.missing_z
        Ez = self.Ez
        Ezz = self.Ezz
        Sxz = data[missing].T @ self.Ez[missing]
        Szz = np.sum(Ezz[missing], axis=0)

        # Initial zte mean/covariance
        self.mu0 = Ez[0].copy() 
        self.Q0 = update_Q0(Ez, Ezz, self.init_state_cov)
        
        self.A = update_A(self.Ez1z[missing_z1], Ezz[missing_z])
        self.C = update_C(Szz, Sxz)
        
        self.b = update_b(Ez[missing_z1], Ez[missing_z], self.A)
        if(self.obs_offset):
            self.d = update_d(data[missing], Ez[missing], self.C)
        
        self.R = update_R(data, Ez[missing], Sxx, Szz, Sxz, self.C, self.d, self.obs_cov)
        self.Q = update_Q(Ez[missing_z], Ez[missing_z1], Szz, Ezz[missing_z], self.Ez1z[missing_z1], self.A, self.b, self.trans_cov)
    
    
    def print_params(self):
        print(f"mu0:{self.mu0}\n")
        print(f"A:{self.A}\n")
        print(f"Q:{self.Q}\n")
        print(f"C:{self.C}\n")
        print(f"d:{self.d}\n")
        print(f"R:{self.R}\n")
        
        
    def print_result(self):
        print(f"loss: {self.loss}")
        print(f'process time: {self.time}')
        
    
    
    def loglikelihood(self, data):
        self.forward(data)
        return self.llh
    
    
    def err(self, data):
        try:
            _, Obs = self.gen()
            return mse(data, Obs)
        except:
            return INF
            
            
    def fit(self, x, missing, obs_offset, k = None, max_iter=30):
        data = x
        self.obs_offset = obs_offset
        self.n, dim_data = data.shape
        self.dim_data = dim_data
        if k is None: 
            self.k = dim_data
        else:
            self.k = k
        return fit_each(self, data, missing, max_iter)
    
    
    def score(self, data, llh=None):
        if llh is None:
            try:
                llh = self.loglikelihood(data)
            except:
                llh = -INF
        return - llh
    
    def fit_mu0(self, data, wtype='uniform', dps=1):
        return nl_fit(self, data, "si", True)
    
    
    def gen(self, n=None, mu0=None):
        k = self.k
        if(n is None): n = self.n
        if(mu0 is None): mu0 = self.mu0
        z = np.zeros((n, k))
        z[0] = mu0.copy()
        
        return _defunc(z, self.A, self.C, self.b, self.d, n) 