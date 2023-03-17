from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from data_cleaning import *
from data_proc import *
import numpy as np


df_raw, df_Xn_raw = load_data()
M = df_Xn_raw.shape[0]
X_raw = df_raw.drop('y',axis = 1)
y = df_raw['y']
std_methods = ['standard','minmax','maxabs','robust'] + list(range(10,81,5))
fill_methods = ['mean', 'median']

fol_name = "Run_1"
random_state = 3
params = {}
lam = np.logspace(-2, 3, 100)
lam_1 = lam[:]
lam_2 = lam[:]
l1 = np.linspace(0,1,100)
n_est = list(range(20,200,20))
m_depth = list(range(3,20,7))
lr = np.logspace(-3,0,10)
m_sm_spl = list(range(2,81,7))
n_neigh = list(range(5,80,5))

params['DummyRegressor'] = {'strategy': ['mean', 'median']}
params['LinearRegression'] = {'fit_intercept': [True, False]}
params['Ridge'] = {'alpha': lam, 'fit_intercept': [True, False]}
params['Lasso'] = {'alpha': lam_1, 'fit_intercept': [True, False], 'max_iter': [1000]}
params['ElasticNet'] = {'alpha': lam_2, 'l1_ratio': l1, 'fit_intercept': [True,False],'max_iter':[1000]}
params['RandomForestRegressor'] = {'n_estimators':n_est, 'max_depth':m_depth,'min_samples_split':m_sm_spl[:]}
params['GradientBoostingRegressor'] = {'n_estimators': n_est, 'learning_rate':lr,'max_depth':m_depth, 'min_samples_split':m_sm_spl}
params['KNeighborsRegressor'] = {'n_neighbors':n_neigh,'n_jobs': [-1],'weights':['uniform', 'distance'],'p':[1,2]}

# Create a list of models
#models = [LinearRegression()]
models = [Ridge(),KNeighborsRegressor(),LinearRegression(),Lasso(),ElasticNet(),RandomForestRegressor(),GradientBoostingRegressor(),DummyRegressor()]
k1 = 10
k2 = 10
best_fill,best_std,best_train,best_name_model,best_param,errors_out,gen_error = twolevelcv(df_raw, k1=k1, k2=k2, models=models,params=params, rs=random_state, fill_methods=fill_methods, std_method = std_methods,gili = fol_name)

names = [0]*k1
models = [0]*k1
ypred = np.zeros((k1,M))

for i in range(k1):
    print(f"Parameter of model {i+1}:")
    names[i] = best_name_model[i][0]
    models[i] = best_name_model[i][1]
    method = best_fill[i]
    std_method = best_std[i]
    train_idx = best_train[i]
    Xtrain,Xtest = transform_data(X_raw.iloc[train_idx,:],df_Xn_raw,fill_method = method,std_method = std_method)
    model = models[i]
    p_ = best_param[i]
    model.set_params(**p_)
    model.fit(Xtrain,y[train_idx])
    print(f"Model: {names}")
    print(f"Fill method: {method}")
    print(f"Std method: {std_method}")
    print(f"Parameters: {model.get_params()}\n")
    ypred[i] = model.predict(Xtest)
  
print("Best fill: {best_fill}")
print(f"Best std: {best_std}")
print(f"Best train_idx1: {best_train}")
print(f"Best mods: {names}")
print(f"Best params: {best_param}")
print(f"Errors with weights: {errors_out}")
print(f"Gen error: {gen_error}")

yfinal = np.mean(ypred,axis = 0)
pp = [gen_error]

np.savetxt("predictions.txt",yfinal)
np.savetxt("gen_error.txt",pp)


