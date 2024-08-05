import numpy as np
from numpy.linalg import cholesky, inv
from numba import njit
from model.utils import l1, l2

ZERO = 1.e-10

@njit(cache=True)
def soft_threshold(y, alpha):
    return np.sign(y) * np.maximum(np.abs(y) - alpha, 0.0)


@njit(cache=True)
def grad(x, snl, s1nl, Q_inv):
    return  Q_inv @ (x @ snl - s1nl)


@njit(cache=True)
def _f(params, Q_chol_inv, y, z, l1_lam, l2_lam, I):
    return np.sum(np.square(Q_chol_inv @ (y - z @ params.T).T))/2 + l1(l1_lam, params - I) + l2(l2_lam, params - I)   


@njit(cache=True)
def _iter(y, z, snl, s1nl, Q_chol_inv, Q_inv, init, Qbz, l_k, l1_lam, l2_lam, k, k_nl, max_iter, ptol):
    x_acc = init.copy()
    x_p = init.copy()
    x_iter = init.copy()
    w = 1.0
    w_p = 1.0
    I = np.eye(k, k+k_nl)
    l1_lam_step = l1_lam/l_k
    l2_lam_step = l2_lam * 2
    f_p = _f(x_p, Q_chol_inv, y, z, l1_lam, l2_lam, I)
    for iter in range(max_iter):
        x_iter = soft_threshold(x_acc - (grad(x_acc, snl, s1nl, Q_inv)  + Qbz + l2_lam_step*(x_acc - I))/l_k - I, l1_lam_step) + I
        f = _f(x_iter, Q_chol_inv, y, z, l1_lam, l2_lam, I)
        conv2 = abs(f - f_p)
        
        if (conv2 < ptol):
            break
        
        if f > f_p:
            w = 1.0
            x_acc = x_p.copy()
        else:
            w_p = w
            w = (1 + np.sqrt(1+4*np.power(w, 2))) / 2
            x_acc = x_iter + (w_p-1.0) / w * (x_iter - x_p)
            x_p = x_iter.copy()
            f_p = f
            
    if f_p < f:
        x_iter = x_p
        f = f_p
    return np.nan_to_num(x_iter, copy=False), f


class PG:
    def __init__(self, ptol, l1_lam, l2_lam, sparse, max_iter=1000):
        self.ptol = ptol
        self.l1_lam = l1_lam
        self.l2_lam = l2_lam
        self.l2_lam_step = l2_lam*2
        self.sparse = sparse
        self.max_iter = max_iter
        

    def fit(self, init, y, b, z, Szznl, Sz1znl, Q, k, k_nl):
        if self.sparse:
            Q_chol_inv = inv(cholesky(Q + ZERO * np.eye(Q.shape[0])))
            Q_inv = Q_chol_inv.T @ Q_chol_inv
            Qbz = Q_inv @ np.outer(b, np.sum(z, 0))
            l1_lam = self.l1_lam
            l2_lam = self.l2_lam
            l2_lam_step = self.l2_lam_step
            l_k = np.sqrt(np.sum(np.power(Q_inv, 2)) * np.sum(np.power(Szznl, 2))) + l2_lam_step
            params, f_s = _iter(y, z, Szznl, Sz1znl, Q_chol_inv, Q_inv, init, Qbz, 
                                l_k, l1_lam, l2_lam, k, k_nl, self.max_iter, self.ptol) 
        else:
            params = Sz1znl @ inv(Szznl)
        
        return params[:, :k], params[:, k:]