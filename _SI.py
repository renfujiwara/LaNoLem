import argparse
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["VECLIB_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["NUMBA_NUM_THREADS"]="1"

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from src.Dysts_dataset import Data
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import multiprocessing

from src import LaNoLem, utils

noise_list=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
seed_list=[1,19,42,1024,3407]


def _iter_LaNoLem(model, data, data_test, y, gt, gt_n,  equation_name, dt, noise_ratio, seed, name, l1_r, l2_r):
    model = LaNoLem(verbose=False, num_works=1)
    model = model.fit(data, fit_type='Robust', l1_r=l1_r, l2_r=l2_r)
    w = np.concatenate((model.C @ (model.A - np.eye(model.k)), model.C @ model.F), axis=1)
    w_aug = np.zeros_like(gt)
    w_aug[:, :w.shape[1]] = w
    model.mu0 = model.mu[-1]
    model.forward(data_test)
    return {"method":name, 
            "Coefficient error":np.linalg.norm(gt-w_aug/dt, ord='fro')/gt_n, 
            "Recall":np.count_nonzero(np.abs(gt*w_aug) > 1.e-10) / (np.count_nonzero(np.abs(gt) > 1.e-10)) * 100, 
            "Prediction error (no noise)":mse(y[1:], model.mu_o[1:]), 
            "Prediction error":mse(data_test[1:], model.mu_o[1:]), 
            "seed":seed, "data_name": equation_name, "noise_ratio": noise_ratio}
        

def _iter(dataset, coef, equation_name, dt, name, l1_r, l2_r):
    k_nl, _ = utils.get_k_nl(coef.shape[0], 4)
    gt = np.zeros((coef.shape[0], coef.shape[0]+k_nl))
    _, j = np.shape(coef)
    gt[:,:j] = coef.copy()
    gt_n = np.linalg.norm(gt, ord='fro')
    model = LaNoLem(verbose=False, num_works=1)
    return [_iter_LaNoLem(model, dataset[noise_ratio][seed][:500], dataset[noise_ratio][seed][499:], dataset[0][seed][499:], gt, gt_n, 
                          equation_name, dt, noise_ratio, seed, name, l1_r, l2_r) for noise_ratio in noise_list for seed in seed_list]
    
def evaluate(args):
    save_path = f'./result/SI/'
    # Dataset = Data(noise_list, seed_list, n = 600)
    Dataset = Data()
    Dataset.make_all_data(noise_list, seed_list, n = 600)
    l = Dataset.get_len()
    print('make data', flush=True)
    if args.num_works == -1:
        num_works = int(os.cpu_count()/4*3)
    else:
        num_works = args.num_works
    
    if args.rtype == 'Lasso':
        name = 'LaNoLem-L'
        l1_r = 1.0
        l2_r = 0.0
    elif args.rtype == 'Naive':
        name = 'LaNoLem-N'
        l1_r = 0.0
        l2_r = 0.0
    else:
        name = 'LaNoLem'
        l1_r = 1.0
        l2_r = 0.5
        
    with ProcessPoolExecutor(max_workers=num_works, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(_iter, Dataset.syn_data[equation_name], Dataset.get_true_coefficients(equation_name), 
                                    equation_name, Dataset.dt, name, l1_r, l2_r): equation_name for equation_name in Dataset.systems_list}
        i = 0
        result=[]
        for future in concurrent.futures.as_completed(futures):
            data_name = futures[future]
            result.extend(future.result())
            print(f'-----------\n{data_name}({i+1}/{l}) end\n-----------', flush=True)
            i += 1
    
    lanolem_df = pd.DataFrame(result)
    os.makedirs(save_path, exist_ok=True)
    lanolem_df.to_csv(f'{save_path}/SI_Errs_method_{args.rtype}_{args.date}.csv',index=False)
        
        
            
def parse_args():
    desc = "hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset_name', default=None, type=str, help='string to identify experiment')
    parser.add_argument('--num_works', default=4, type=int, help='num process')
    parser.add_argument('--date', type=str, help='num process')
    parser.add_argument('--rtype', default='Adaptive', type=str, help='regularizer')
    return parser.parse_args()

if __name__ == "__main__":
    print(os.getpid())
    args = parse_args()
    if args is None:
        exit()
    print('main_start', flush=True)
    evaluate(args)
    print('main_end', flush=True)
