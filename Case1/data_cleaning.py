import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

def load_data():
    df = pd.read_csv(r"case1Data.txt", sep=', ', engine='python')
    df_Xn = pd.read_csv(r"case1Data_Xnew.txt",  sep=', ', engine='python')
    return df, df_Xn

def standarize(df, method):
    if method == 'standard':
        df = (df - df.mean()) / df.std()
    elif method == 'minmax':
        df = (df - df.min()) / (df.max() - df.min())
    elif method == 'maxabs':
        df = df / df.abs().max()
    elif method == 'robust':
        df = (df - df.median()) / (df.quantile(0.75) - df.quantile(0.25))
    else:
        raise ValueError('Invalid standardization method')
    return df

def transform_data(df, fill_method, std_method):
    # Encode categorical variables
    imputer_numeric = SimpleImputer(missing_values=np.nan, strategy=fill_method)
    imputer_categorical = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    df_num = df._get_numeric_data()
    df_num_std = standarize(df_num, std_method)
    df_num_final = imputer_numeric.fit_transform(df_num_std)

    df_cat = df.drop(df_num.columns, axis=1)

    df_cat_encoded = MultiColumnLabelEncoder(columns=df_cat.columns).fit_transform(df_cat)
    mask = df_cat.isnull()
    df_cat_temp = df_cat_encoded.where(~mask)
    df_cat = imputer_categorical.fit_transform(df_cat_temp)

    df_trans = pd.DataFrame(np.concatenate((df_num_final, df_cat), axis=1), columns=df.columns)

    X = df_trans.drop(columns=['y'])
    y = df['y']
    
    return X, y




    
