import yaml
import numpy as np
import pandas as pd
import scipy
from scipy.integrate import odeint
 
 
def vanderPol(var, t, u, B):
    dx = var[1]
    dv = -u*(var[0]*var[0]-1)*var[1]-B*var[0]
    return dx, dv


def cubic(var, t, a, b, c, d):
    dx = a*var[0]**3 + b*var[1]**3
    dy = c*var[0]**3 + d*var[1]**3
    return dx, dy


def hopf(var, t, u, o, A):
    dx = u*var[0]+o*var[1]-A*var[0]*(var[0]**2+var[1]**2)
    dy = -o*var[0]+u*var[1]-A*var[1]*(var[0]**2+var[1]**2)
    return dx, dy


def halvorsen(var, t, a, b):
    dx = -a*var[0] -b*var[1] - b*var[2] - var[1]**2
    dy = -b*var[0] -a*var[1] - b*var[2] - var[2]**2
    dz = -b*var[0] -b*var[1] - a*var[2] - var[0]**2
    return dx, dy, dz


def rucklidge(var, t, k, lam):
    dx = - k*var[0] + lam * var[1] - var[1]*var[2]
    dy = var[0]
    dz = - var[2] + var[1]**2
    return dx, dy, dz




def make_data(data_name, noise_ratio = None, noise_mean=None, noise_std=None, setting=None, random_state=None,
              latent = False):
    
    
    if(data_name == 'outdoor'):
        with open(f'./settings/format.yaml') as file:
            setting = yaml.safe_load(file)
        setting['data_name'] = data_name
        plot_args ={}
        if random_state is None:
            random_state = setting['random_state']
        df = pd.read_csv("./dataset/googletrends/outdoor_7.csv")
        df1 = df.drop(['date'], axis=1)
        df1 = scipy.stats.zscore(df1)
        df1['date'] = pd.to_datetime(df['date'])
        df = df1[(df1['date'] <= "2022-12-25") & (df['date'] >= "2010-04-05")].drop('date',axis=1).copy()
        for name in df.columns:
            if name == 'date': continue
            df[name] = df1[name].rolling(4).mean()
        df['date'] = df1['date']
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop(['date'], axis=1, inplace=True)
        setting['yticklabels'] = df.columns.values
        return df.values, setting
    
    elif(data_name == 'ship'):
        with open(f'./settings/format.yaml') as file:
            setting = yaml.safe_load(file)
        setting['data_name'] = data_name
        plot_args ={}
        if random_state is None:
            random_state = setting['random_state']
            
        df = pd.read_csv('./dataset/Ship/patrol_ship_routine/processed/ship_ind/20190805-095929.csv', index_col=0)
        df.drop(['n'], axis=1, inplace=True)
        df = scipy.stats.zscore(df)
        for name in df.columns:
            df[name] = df[name].rolling(60).mean()
        df.dropna(inplace=True)
        setting['yticklabels'] = df.columns.values
        return df.values, setting
            
        
    if setting is None:
        with open(f'./settings/SI/{data_name}.yaml') as file:
            setting = yaml.safe_load(file)
        setting['data_name'] = data_name
    plot_args ={}
    if random_state is None:
        random_state = setting['random_state']
    
    
    num_steps = setting['num_steps']
    dt = setting['dt']
    t_list = np.linspace(0.0, num_steps*dt, num_steps)
    
    
    init = setting['init']    
    plot_args['data_name'] = data_name
    plot_args['data'] = None
    plot_args['fig_type'] = 'pdf'
    plot_args['dir_path'] = f'./result/{data_name}'
            
    if(data_name == 'cubic'):
        a = setting['a']
        b = setting['b']
        c = setting['c']
        d = setting['d']
        gt = np.array([[0,0,0,0,0,a,0,0,b], [0,0,0,0,0,c,0,0,d]])
        data_no_noise = odeint(cubic, init, t_list, args=(a, b, c, d))
        
    elif(data_name == 'vanderpol'):
        u = setting['u']
        B = setting['B']
        gt = np.array([[0,1,0,0,0,0,0,0,0], [-B,u,0,0,0,0,-u,0,0]])
        data_no_noise = odeint(vanderPol, init, t_list, args=(u, B))
        
    elif(data_name == 'hopf'):
        u = setting['u']
        o = setting['o']
        A = setting['A']
        gt = np.array([[u,o,0,0,0,-A,0,-A,0], [-o,u,0,0,0,0,-A,0,-A]])
        data_no_noise = odeint(hopf, init, t_list, args=(u, o, A))
        
    elif(data_name == 'halvorsen'):
        a = setting['a']
        b = 4
        gt = np.array([[-a,-b,-b,0,0,0,-1,0,0], [-b,-a,-b,0,0,0,0,0,-1], [-b,-b,-a,-1,0,0.0,0,0,0]])
        data_no_noise = odeint(halvorsen, init, t_list, args=(a, b))
        
    elif(data_name == 'rucklidge'):
        k = setting['k']
        lam = setting['lam']
        gt = np.array([[-k,lam,0,0,0,0,0,-1,0], [1,0,0,0,0,0,0,0,0], [0,0,-1,0,0,0,1,0,0]])
        data_no_noise = odeint(rucklidge, init, t_list, args=(k, lam))
        
    setting['gt'] = gt
    
    
    if noise_ratio is not None:
        if random_state is None:
            random_state = setting['random_state']
        else:
            setting['random_state'] = random_state
        
        if noise_mean is None:
            noise_mean = setting['noise_mean']

        if noise_std is None:
            noise_std = setting['noise_std']
            
        data_norm = np.linalg.norm(data_no_noise, axis=0)
        rand = np.random.default_rng(seed=random_state)
        noise = rand.normal(0, 1.0, size=(num_steps, data_no_noise.shape[1]))
        noise_norm = np.linalg.norm(noise, axis=0)
        data = data_no_noise + noise * (data_norm * noise_ratio) / noise_norm
        return data, setting
    
    else:
        return data_no_noise, setting
