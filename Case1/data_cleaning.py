import pandas as pd
def load_data()
    df = pd.read_csv(r"case1Data.txt", sep=', ', engine='python')
    y = df.y
    X_num = df.iloc[:, 1:96]
    df_Xn = pd.read_csv(r"case1Data_Xnew.txt",  sep=', ', engine='python')