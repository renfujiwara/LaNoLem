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
    return np.dot(np.sum(Ez1z, axis=0), inv(np.sum(Ezz[:-1], axis=0)))


def update_C(Szz, Sxz):
    return np.dot(Sxz, inv(Szz))


def update_b(Ez, A):
    return np.mean(Ez[1:] - Ez[:-1] @ A.T, axis=0)


def update_d(data, Ez, C):
    return np.mean(data - Ez @ C.T, axis=0)


def update_Q(Ez, Szz, Ezz, Ez1z, A, b, covariance_type):
    Sz1z = np.sum(Ez1z, axis=0)
    val = A @ (Sz1z.T - np.einsum('ij,l->jl',  Ez[:-1], b)) + np.einsum('ij,l->jl',  Ez[1:], b)
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


def fit_each(model_org, data, max_iter):
    model = deepcopy(model_org)
    result = {"md":None, "err":None, "loss":None}
    
    model.init_params(data)
    model, llh = model.initialize(data)
    model.iter = 0
    history_llh = [llh]
    conv = 'max'
    Sxx = data.T @ data
    
    best_model = deepcopy(model)
    for iteration in range(max_iter):
        model.iter = iteration
        try:
        # inference-step
            model.forward(data)
            model.backward()
            
            # learning-step
            model.solve(data, Sxx)

            loss = model.score(data)
            model.loss = loss
            llh = model.llh
        except (np.linalg.LinAlgError, FloatingPointError):
            conv = 'except'
            break
        
        history_llh.append(llh)
        
        if history_llh[-1] < history_llh[-2]:
            conv = 'conv1'
            break
        else:
            best_model = deepcopy(model)
            
    model = best_model
    model.conv = conv
    return model


@njit(cache=True)
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


@njit(cache=True)
def _forward(data, mu0, A, b, Q, Q0, C, R, d, k):
    n, dim_data = data.shape
    Ih = np.eye(k)
    mu = np.empty((n, k))
    mu_t = np.zeros((n, k))
    mu_o = np.empty((n, dim_data))
    sgm = np.empty((n, dim_data, dim_data))
    V = np.empty((n, k, k))
    P = np.zeros((n, k, k))
    
    mu_t[0] = mu0.copy()
    mu[0], mu_o[0], sgm[0], V[0], llh = _filter(mu0, data[0], Q0, C, R, d, Ih)
    
    for t in range(1, len(data)):
        mu_t[t] = A @ mu[t-1] + b
        P[t-1] = A @ V[t - 1] @ A.T + Q
        mu[t], mu_o[t], sgm[t], V[t], llh_t = _filter(mu_t[t], data[t], P[t-1], C, R, d, Ih)
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
        J = V[t] @ A[t].T @ inv(P[t])
        Ez[t] = mu[t] + J @ (Ez[t+1] - mu_t[t+1])
        Ez1z[t] = Vhat @ J.T + np.outer(Ez[t + 1], Ez[t])
        Vhat = V[t] + J @ (Vhat - P[t]) @ J.T
        Ezz[t] = Vhat + np.outer(Ez[t], Ez[t])
    
    return Ez, Ezz, Ez1z


class LDS:
    def __init__(self, data, fn, ma, init_state_cov = "full", trans_cov = "full", obs_cov = "full",
                 print_log = False, verbose=False, random_state = 42):
        
        self.init_state_cov = init_state_cov
        self.trans_cov = trans_cov
        self.obs_cov = obs_cov
        self.print_log = print_log
        self.verbose = verbose
        self.random_state = random_state
        self.fn = fn
        self.ma = ma
    
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
        
    
    def initialize(self, data):
        self.n, self.dim_data = np.shape(data)
        self.loss = self.score(data)
        llh = self.llh

        return self, llh
    
    

    def forward(self, data):
        self.mu, self.mu_t, self.mu_o, self.sgm, self.V, self.P, self.llh = _forward(data, self.mu0, self.A, self.b, self.Q, 
                                                                                     self.Q0, self.C, self.R, self.d, self.k)
            
        
    def backward(self):
        self.Ez, self.Ezz, self.Ez1z = _backward(self.mu, self.mu_t, self.V, self.P, self.A, self.k)
        


    def solve(self, data, Sxx):
        Ez = self.Ez
        Ezz = self.Ezz
        Sxz = np.einsum('ij,il->jl', data, self.Ez)
        Szz = np.sum(Ezz, axis=0)

        # Initial zte mean/covariance
        self.mu0 = Ez[0].copy() 
        self.Q0 = update_Q0(Ez, Ezz, self.init_state_cov)
        
        self.A = update_A(self.Ez1z, Ezz)
        
        self.C = update_C(Szz, Sxz)
        
        self.b = update_b(Ez, self.A)
        if(self.obs_offset):
            self.d = update_d(data, Ez, self.C)
        
        self.R = update_R(data, Ez, Sxx, Szz, Sxz, self.C, self.d, self.obs_cov)
        self.Q = update_Q(Ez, Szz, Ezz, self.Ez1z, self.A, self.b, self.trans_cov)
    
    
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
            
            
    def fit(self, x, obs_offset, k = None, max_iter=30):
        data = x
        self.obs_offset = obs_offset
        self.n, dim_data = data.shape
        self.dim_data = dim_data
        if k is None: 
            self.k = dim_data
        else:
            self.k = k
        return fit_each(self, data, max_iter)
    
    
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