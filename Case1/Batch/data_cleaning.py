import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def rem_cat(df):
    df['C_ 1'].replace(['I','H','K','G','J'],[0,1,2,3,4],inplace=True)
    df['C_ 2'].replace(['I','H','K','G','J'],[0,1,2,3,4],inplace=True)
    df['C_ 3'].replace(['I','H','K','G','J'],[0,1,2,3,4],inplace=True)
    df['C_ 4'].replace(['I','H','K','G','J'],[0,1,2,3,4],inplace=True)
    df['C_ 5'].replace(['I','H','K','G','J'],[0,1,2,3,4],inplace=True)


def load_data():
    X = pd.read_csv(r"case1Data.txt", sep=', ', engine='python')
    X_Xn = pd.read_csv(r"case1Data_Xnew.txt",  sep=', ', engine='python')
    return X, X_Xn

def transform_data(df_train, df_test, fill_method, std_method):
    # Encode categorical variables
    imputer_numeric = SimpleImputer(missing_values=np.nan, strategy=fill_method)
    imputer_categorical = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    
    df_num = df_train._get_numeric_data()
    df_tt_num = df_test._get_numeric_data()
    
    # standarize
    if std_method == 'standard':
        a = df_num.mean()
        b = df_num.std()
        df_num_std = (df_num - a) / b
        df_tt_num_std = (df_tt_num - a) / b
    elif std_method == 'minmax':
        a = df_num.min()
        b = df_num.max()
        df_num_std = (df_num - a) / (b - a)
        df_tt_num_std = (df_tt_num - a) / b
    elif std_method == 'maxabs':
        a = df_num.abs().max()
        df_num_std = df_num / a
        df_tt_num_std = df_tt_num / a
    elif std_method == 'robust':
        a = df_num.median()
        b = df_num.quantile(0.75)
        c = df_num.quantile(0.25)
        df_num_std = (df_num - a) / (b - c)
        df_tt_num_std = (df_tt_num - a) / (b-c)
    elif (type(std_method) is int) and (std_method > 0):
        a = df_num.mean()
        b = df_num.std()
        df_num_std = (df_num - a) / b
        df_tt_num_std = (df_tt_num - a) / b
        
        aux1 = imputer_numeric.fit_transform(df_num_std)
        aux1_tt = imputer_numeric.transform(df_tt_num_std)
        
        pca = PCA(n_components = std_method)
        df_num_std = pca.fit_transform(aux1)
        df_tt_num_std = pca.transform(aux1_tt)
        
        df_cat = df_train.drop(df_num.columns, axis=1)
        df_tt_cat = df_test.drop(df_tt_num, axis =1)
        rem_cat(df_cat)
        rem_cat(df_tt_cat)

        df_cat = imputer_categorical.fit_transform(df_cat)
        df_tt_cat = imputer_categorical.transform(df_tt_cat)

        df_trans = pd.DataFrame(np.concatenate([df_num_std, df_cat], axis=1))#, columns=df_train.columns)
        df_tt_trans = pd.DataFrame(np.concatenate([df_tt_num_std, df_tt_cat], axis=1))#, columns=df_test.columns)
       
        return df_trans.values,df_tt_trans.values
    else:
        raise ValueError('Invalid standardization method')
    
    # impute numeric values
    df_num_final = imputer_numeric.fit_transform(df_num_std)
    df_tt_num_final = imputer_numeric.transform(df_tt_num_std)

    df_cat = df_train.drop(df_num.columns, axis=1)
    df_tt_cat = df_test.drop(df_tt_num, axis =1)
    rem_cat(df_cat)
    rem_cat(df_tt_cat)

    df_cat = imputer_categorical.fit_transform(df_cat)
    df_tt_cat = imputer_categorical.transform(df_tt_cat)

    df_trans = pd.DataFrame(np.concatenate([df_num_final, df_cat], axis=1))#, columns=df_train.columns)
    df_tt_trans = pd.DataFrame(np.concatenate([df_tt_num_final, df_tt_cat], axis=1))#, columns=df_test.columns)

    return df_trans.values,df_tt_trans.values


    
