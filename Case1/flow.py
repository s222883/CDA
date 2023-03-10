from data_cleaning import *

df_raw, df_Xn_raw = load_data()


fill_methods = ['mean'] #, 'median']
std_methods = ['standard'] #, 'minmax', 'maxabs', 'robust']
models= ['logistic'] #, 'svm', 'knn', 'tree', 'forest', 'boosting']

for fill_method in fill_methods:
    for std_method in std_methods:
        print(fill_method)
        X, y = transform_data(df_raw, fill_method, std_method)
        X.to_csv('df.csv', index=False)
        y.to_csv('y.csv', index=False)
        # print('Fill method: {}, Standardization method: {}'.format(fill_method, std_method))
        # print('X shape: {}, y shape: {}'.format(X.shape, y.shape))
        # print('Xn shape: {}'.format(Xn.shape))
