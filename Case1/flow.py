from data_cleaning import *

df_raw, df_Xn_raw = load_data()

fill_methods = ['mean', 'median', 'most_frequent']
std_methods = ['standard', 'minmax', 'maxabs', 'robust']

run_fn()
