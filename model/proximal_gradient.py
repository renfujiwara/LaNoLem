import numpy as np
from numpy.linalg import inv
from numba import njit

MAX_RESTART = 5
ZERO = 1.e-20
INF = 1.e+20

@njit('f8[:,:](f8[:,:],f8[:,:])')
def soft_threshold(y, alpha):
    return np.sign(y) * np.maximum(np.abs(y) - alpha, 0.0)


@njit('f8(f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
def func(y, Q_inv, z, aug_z):
    f = 0
    diff = z[1:] - aug_z[:-1] @ y.T
    for t in range(len(diff)):
        f += diff[t] @ Q_inv @ diff[t]
    return f/2


@njit('f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
def grad(x, snl, s1nl, Q_inv):
    return  Q_inv @ (x @ snl - s1nl)


@njit('f8(f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
def F(y, Q_inv, z, aug_z, I, lams):
    return func(y, Q_inv, z, aug_z) + np.sum(np.abs((y - I)) * lams)
    

@njit('Tuple((f8[:,:],f8))(f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],i8,i8,i8,f8)')
def _iter(snl, s1nl, Q_inv, init, z, aug_z, lams, k, k_nl, max_iter, ptol):
    w=1.0
    I = np.eye(k, k+k_nl)
    x_iter =  np.zeros((k, k+k_nl))
    y = init.copy()
    l_k = np.sqrt(np.sum(np.power(Q_inv, 2)) * np.sum(np.power(snl, 2)))
    restart_num = 0
    n = len(z)
    for iter in range(max_iter):
        if restart_num > MAX_RESTART:
            restart_num = 0
            w = 1.0
            continue
        
        grad_y = grad(y, snl, s1nl, Q_inv) 
        y_iter = y - grad_y/l_k - I
        x_iter = soft_threshold(y_iter, lams/l_k)
        x_iter += I
        
        norm_y = np.sqrt(np.sum(np.power(y, 2)))
        norm_x_iter = np.sqrt(np.sum(np.power(x_iter-y, 2)))
        if norm_y < ZERO:
            if norm_x_iter < ptol:
                break
        elif norm_x_iter/norm_y < ptol:
            break
        
        w_p = w
        w = (1 + np.sqrt(1+4*np.power(w, 2))) / 2
        y = x_iter + (w_p-1) * (x_iter - y) / w
        restart_num += 1
        
    _coef = np.nan_to_num(x_iter, copy=False)
    return _coef, F(x_iter, Q_inv, z, aug_z, I, lams)


class PG:
    def __init__(self, ptol, lams, max_iter=1000, min_covar=1.e-10):
        self.ptol = ptol
        self.lams = lams
        self.max_iter = max_iter
        self.min_covar = min_covar

    def fit(self, AF, z, aug_z, Szznl, Sz1znl, Q, k, k_nl):
        I = np.eye(k, k+k_nl)
        S_inv = inv(Szznl)
        Q_inv = inv(Q)
        best_params = Sz1znl @ inv(Szznl)
        F_x_iter = F(best_params, Q_inv, z, aug_z, I, self.lams)
        
        for init in  [AF, Sz1znl @ S_inv]:
            params, F_x = _iter(Szznl, Sz1znl, Q_inv, init, z, aug_z, self.lams, k, k_nl, self.max_iter, self.ptol)
            if F_x_iter > F_x:
                best_params = params
                F_x_iter = F_x
            
        return best_params[:,:k], best_params[:,k:]