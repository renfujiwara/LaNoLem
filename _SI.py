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
import pysindy as ps
from pysindy.optimizers import SSR, STLSQ, MIOSR
from pysindy.feature_library import PolynomialLibrary
import glob
import yaml

from model import NLDS, make_data, plot_result, utils

def identify(args):
    if args.dataset_name is None:
        for file in glob.glob(os.path.join(f'./settings/SI/', "*.yaml")):
            with open(file) as f:
                setting = yaml.safe_load(f)
            dataset_name = setting['data_name']
            print(f'data:{dataset_name}')
            data, setting = make_data(dataset_name, setting=setting, noise_ratio=0.1)
            model = NLDS(num_works=args.num_works)
            model.random_state = setting['random_state']
            model = model.fit(data, fit_type='Robust')
            plot_result(model, data, setting, fsize=3.3)
        
    else:
        dataset_name = args.dataset_name
        data, setting = make_data(data_name=dataset_name, noise_ratio=0.1)
        model = NLDS(num_works=args.num_works)
        model.random_state = setting['random_state']
        model = model.fit(data, fit_type='Robust')
        plot_result(model, data, setting, fsize=3.3)
    
    
def evaluate(args):
    save_path = f'./result/SI/'
    noise_list=[0.05, 0.25, 0.5]
    seed_list=[1,19,42,1024,3407]
    mse_dict={}
    df_list=[]
    sindy_opt = {"STLSQ":STLSQ,"MIOSR":MIOSR,"SSR":SSR}
    files = glob.glob(os.path.join(f'./settings/SI/', "*.yaml"))
    l = len(files)
    for i, file in enumerate(files):
        with open(file) as f:
            setting = yaml.safe_load(f)
        data_name = setting['data_name']
        data_no_noise, setting = make_data(data_name)
        print(f'-----------')
        print(f'{data_name}({i}/{l})')
        print(f'-----------')
        # print(coef)
        data_norm = np.linalg.norm(data_no_noise, ord='fro')
        k_nl, _ = utils.get_k_nl(data_no_noise.shape[1], 4)
        gt = np.zeros((data_no_noise.shape[1], data_no_noise.shape[1]+k_nl))
        coef = setting['gt'] 
        dt = setting['dt']
        _, j = np.shape(coef)
        gt[:,:j] = coef.copy()
        gt_n = np.linalg.norm(gt, ord='fro')
        for noise_ratio in noise_list:
            print(f'-----------')
            print(f'noise_ratio:{noise_ratio}')
            print(f'-----------')
            for seed in seed_list:
                mse_dict={}
                rand = np.random.default_rng(seed=seed)
                noise = rand.normal(0, 1.0, size=(len(data_no_noise), data_no_noise.shape[1]))
                noise_norm = np.linalg.norm(noise, ord='fro')
                data = data_no_noise + noise * (data_norm * noise_ratio) / noise_norm
                
                model = NLDS(dt = dt)
                model.random_state = seed
                model = model.fit(data, fit_type='Robust')
                w = np.concatenate((model.C @ (model.A - np.eye(model.k)), model.C @ model.F), axis=1) 
                w_aug = np.zeros_like(gt)
                w_aug[:, :w.shape[1]] = w
                mse_dict['ScatTerm'] = np.linalg.norm(gt-w_aug/dt, ord='fro')/gt_n

                for opt in ["STLSQ", "MIOSR", "SSR"]:
                    best_aic = float('inf')
                    best_err = float('inf')
                    for alpha in [0, 1.e-5, 1.e-3, 1.e-2, 0.05, 0.2]:
                        for dim in [2,3,4]:
                            sindy = ps.SINDy(optimizer=sindy_opt[opt](alpha=alpha), feature_library=PolynomialLibrary(degree=dim))
                            sindy = sindy.fit(data, t = dt)
                            k = np.count_nonzero(np.abs(sindy.coefficients()[:,1:]) > 1e-8)
                            aic = data.size*np.log(sindy.score(data, t=dt, metric=mse))+2*k+2*(k+1)*(k+2)/(data.size-k-2)
                            if best_aic > aic:
                                best_aic = aic
                                w_aug = np.zeros_like(gt)
                                w = sindy.coefficients()[:,1:]
                                w_aug[:, :w.shape[1]] = w
                                best_err = np.linalg.norm(gt-w_aug, ord='fro')/gt_n
                    mse_dict[opt] = best_err
                print(mse_dict)
                df = pd.DataFrame.from_dict(mse_dict, orient="index", columns=["mse"])
                df['seed'] = seed
                df['data_name'] = setting['data_name']
                df['noise_ratio'] = noise_ratio
                df.reset_index(inplace= True)
                df = df.rename(columns={'index':'method'})
                df_list.append(df)
            
    df = pd.concat(df_list).reset_index(drop=True)
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/SI_Errs_all_0519.csv',index=False)
            
def parse_args():
    desc = "hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset_name', default=None, type=str, help='string to identify experiment')
    parser.add_argument('--num_works', default=4, type=int, help='num process')
    parser.add_argument('--evaluate', default=0, type=int, help='num process')
    return parser.parse_args()

if __name__ == "__main__":
    print(os.getpid())
    args = parse_args()
    if args is None:
        exit()
    print('main_start', flush=True)
    if args.evaluate == 0:
        identify(args)
    else:
        evaluate(args)

    print('main_end', flush=True)