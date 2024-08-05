import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import pysindy as ps
from pysindy.optimizers import SSR, STLSQ, MIOSR
from pysindy.feature_library import PolynomialLibrary
from model.Dysts_dataset import Data

from model import utils

noise_list=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
seed_list=[1,19,42,1024,3407]
sindy_opt = {"STLSQ":STLSQ,"MIOSR":MIOSR,"SSR":SSR}


def _iter_syndy(data_all, y, coef, opt, dt, seed, equation_name, noise_ratio):
    aic=[]
    SINDys=[]
    data = data_all[:500]
    k_nl, _ = utils.get_k_nl(coef.shape[0], 4)
    gt = np.zeros((coef.shape[0], coef.shape[0]+k_nl))
    _, j = np.shape(coef)
    gt[:,:j] = coef.copy()
    gt_n = np.linalg.norm(gt, ord='fro')
    n = data.shape[0]
    for dim in [4, 3, 2]:
        for alpha in [0.2, 0.05, 1.e-2, 1.e-3, 1.e-5, 0.0]:
            sindy = ps.SINDy(optimizer=sindy_opt[opt](alpha=alpha), feature_library=PolynomialLibrary(library_ensemble=None, degree=dim))
            sindy = sindy.fit(data, t = dt)
            k = sindy.complexity
            aic.append(n*np.log(sindy.score(data, t=dt, metric=mse))+2*k+2*(k+1)*(k+2)/(n-k-2))
            SINDys.append(sindy)
    best_i = np.argmin(aic)
    sindy = SINDys[best_i]
    w_aug = np.zeros_like(gt)
    w = sindy.coefficients()[:,1:]
    w_aug[:, :w.shape[1]] = w
    x_pred = sindy.predict(data_all[499:-1]) * dt + data_all[499:-1]
    return {"method":opt, 
            "Coefficient error":np.linalg.norm(gt-w_aug, ord='fro')/gt_n, 
            "Recall":np.count_nonzero(np.abs(gt*w_aug) > 1.e-10) / (np.count_nonzero(np.abs(gt) > 1.e-10)) * 100, 
            "Prediction error (no noise)":mse(y[500:], x_pred), 
            "Prediction error":mse(data_all[500:], x_pred),
            "seed":seed, "data_name": equation_name, "noise_ratio": noise_ratio}
        
        
def evaluate(args):
    save_path = f'./result/SI/'
    Dataset = Data(noise_list, seed_list, n = 600)
    # l = Dataset.get_len()
    print('make data', flush=True)
    result_list_sindy = [_iter_syndy(Dataset.syn_data[equation_name][noise_ratio][seed], 
                                        Dataset.syn_data[equation_name][0][seed], 
                                        Dataset.get_true_coefficients(equation_name), 
                                        args.rtype, Dataset.dt, seed, 
                                        equation_name, noise_ratio) for equation_name in Dataset.systems_list for noise_ratio in noise_list for seed in seed_list]
            
    syndy_df = pd.DataFrame(result_list_sindy)
    os.makedirs(save_path, exist_ok=True)
    syndy_df.to_csv(f'{save_path}/SI_Errs_{args.rtype}_{args.date}.csv',index=False)
        
        
            
def parse_args():
    desc = "hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset_name', default=None, type=str, help='string to identify experiment')
    parser.add_argument('--num_works', default=4, type=int, help='num process')
    parser.add_argument('--date', type=str, help='num process')
    parser.add_argument('--rtype', default='MIOSR', type=str, help='regularizer')
    return parser.parse_args()

if __name__ == "__main__":
    print(os.getpid())
    args = parse_args()
    if args is None:
        exit()
    print('main_start', flush=True)
    evaluate(args)
    print('main_end', flush=True)
