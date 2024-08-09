import yaml
import numpy as np
import pandas as pd
import scipy




def make_data(data_name):    
    if(data_name == 'outdoor'):
        with open(f'./settings/format.yaml') as file:
            setting = yaml.safe_load(file)
        setting['data_name'] = data_name
        plot_args ={}
        if random_state is None:
            random_state = setting['random_state']
        df = pd.read_csv("./dataset/googletrends/outdoor_7.csv")
        df1 = df.drop(['date', 'Diving'], axis=1)
        df1 = scipy.stats.zscore(df1)
        df1['date'] = pd.to_datetime(df['date'])
        df = df1[(df1['date'] <= "2022-12-25") & (df['date'] >= "2010-04-05")].drop('date',axis=1).copy()
        for name in df.columns:
            if name == 'date': continue
            df[name] = df1[name].rolling(10).mean()
        df['date'] = df1['date']
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop(['date'], axis=1, inplace=True)
        setting['yticklabels'] = df.columns.values
        data = df.values
        shift_amount = -np.min(data) + 0.1
        
        return data + shift_amount, setting
    