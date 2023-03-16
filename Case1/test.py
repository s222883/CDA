import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from data_cleaning import *
from sklearn.metrics import mean_squared_error as mse


df_raw, df_Xn_raw = load_data()

fill_methods = ['mean', 'median', 'most_frequent']
std_methods = ['standard', 'minmax', 'maxabs', 'robust']

X = transform_data(df_raw, fill_methods[0], std_methods[0])[0].to_numpy()

y = df_raw['y'].to_numpy()

"""
pca = decomposition.PCA(n_components = 40)
pca.fit(X)
X = pca.transform(X)
"""
num_try = 20000

al = np.zeros(num_try)

al = np.logspace(-5,3,num = num_try)

k1 = 5
k2 = 5

cv = KFold(k1,random_state = 123,shuffle = True)

err_gen = np.zeros(k1)
params = np.zeros(k1)

num_neigh = np.array(list(range(1,1001)))

error = np.zeros((k2,len(al)))

for i,(train_idx1,test_idx1) in enumerate(cv.split(X,y)):
	cv_val = KFold(k2)
	error *= 0
	for l,(train_idx2,test_idx2) in enumerate(cv_val.split(X[train_idx1,:],y[train_idx1])):
		XTrain = X[train_idx2,:]
		yTrain = y[train_idx2]
		XTest = X[test_idx2,:]
		yTest = y[test_idx2]
		for j in range(len(al)):
			model = Ridge(alpha=al[j])
			model.fit(XTrain,yTrain)
			pred = model.predict(XTest)
			error[l,j] = mse(yTest,pred,squared = False)
	err = np.mean(error,axis = 0)
	params[i] = al[np.argmin(err)]
	model = Ridge(alpha = params[i])
	model.fit(X[train_idx1,:],y[train_idx1])
	pred = model.predict(X[test_idx1,:])
	err_gen[i] = mse(y[test_idx1],pred,squared = False)

print(params)
print(err_gen)


