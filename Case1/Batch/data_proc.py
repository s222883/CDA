from data_cleaning import *
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
from sklearn.model_selection import ParameterGrid
from scipy.stats import pearsonr
from tqdm import tqdm
import itertools
import warnings
warnings.filterwarnings('ignore')

def twolevelcv(df, k1: int, k2: int, models: list, params: dict, rs: int, fill_methods: list, std_method: list,gili: str):
    
    if os.path.exists(gili):
        raise ValueError(f"folder {gili} already exists, we need it to not exist")
        return None
    
    names = [type(m).__name__ for m in models]
    kf1 = KFold(k1, shuffle = True, random_state=rs)
    X_raw = df.drop('y', axis=1)
    y = df['y']
    
    combs = list(itertools.product(fill_methods,std_method,zip(names,models)))
    it_combs = len(combs)
    N = X_raw.shape[0]
    
    best_fill = [0]*k1
    best_std = [0]*k1
    best_name_model = [0]*k1
    errors_out = [0]*k1
    best_param = [0]*k1
    best_train = [0]*k1
    
    # first level split
    for z,(train_idx1, test_idx1) in enumerate(kf1.split(X_raw, y)):
        print(f"Computing KFold {z+1}/{k1}")
        kf2 = KFold(k2, shuffle = True, random_state=rs+z+1)
        err = [[] for pp in range(it_combs)]
        # second level split
        for t,(train_idx2, test_idx2) in enumerate(kf2.split(X_raw.iloc[train_idx1, :], y[train_idx1])):
            for i in range(it_combs):
                method = combs[i][0]
                std_method = combs[i][1]
                name,model = combs[i][2]
                X_train,X_test = transform_data(X_raw.iloc[train_idx2, :],X_raw.iloc[test_idx2, :], fill_method=method, std_method=std_method)
                y_train = y[train_idx2]
                y_test = y[test_idx2]
                grid = list(ParameterGrid(params[name]))
                n_p = len(grid)
                if err[i] == []:
                    err[i] = [[] for pp in range(n_p)]
                for j in range(n_p):
                    p_ = grid[j]
                    model = model.set_params(**p_)
                    # train the model
                    model.fit(X_train, y_train)
                    # evaluate performance
                    pred2_test = model.predict(X_test)
                    error = mse(pred2_test, y_test,squared = False)
                    err[i][j].append(error*len(test_idx2)/len(train_idx1))
        # inner cv has finished, choose model and param
        best_err = np.inf
        i_best = None
        j_best = None
        for i in range(it_combs):
            for j in range(len(err[i])):
                method = combs[i][0]
                std_method = combs[i][1]
                name = combs[i][2][0]
                fol = gili + '/' + 'iter_'+ str(z) + '/'+ method + '/' + str(std_method) + '/' + name
                
                if not os.path.exists(fol):
                    os.makedirs(fol)
                fname = fol + '/' + str(j) + '.txt'
                pp = np.array(err[i][j])*90.0/9.0
                np.savetxt(fname,pp,header = str(grid[j]))                
                
                aux = np.sum(err[i][j])
                if aux < best_err:
                    i_best = i
                    j_best = j
                    best_err = aux
        method = combs[i_best][0]
        std_method = combs[i_best][1]
        name,model = combs[i_best][2]
        grid = list(ParameterGrid(params[name]))
        p_ = grid[j_best]
        model = model.set_params(**p_)

        X_tr,X_te = transform_data(X_raw.iloc[train_idx1, :], X_raw.iloc[test_idx1, :], fill_method=method, std_method=std_method)
        y_te = y[test_idx1]
        y_tr = y[train_idx1]
        model.fit(X_tr,y_tr)
        pred = model.predict(X_te)
        error = mse(pred,y_te,squared = False)
        
        best_fill[z] = method
        best_std[z] = std_method
        best_name_model[z] = (name,model)
        print(f"fill method: {method}, std_method: {std_method}, model: {name} with parameter: {p_}")
        print(f"error: {error}\n")
        errors_out[z] = error*len(test_idx1)/N
        best_param[z] = p_
        best_train[z] = train_idx1
    
    gen_error = np.sum(errors_out)
    
    return best_fill,best_std,best_train,best_name_model,best_param,errors_out,gen_error
