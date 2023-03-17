
def twolevelcv(df, k1: int, k2: int, models: list, params: dict, rs: int, fill_methods: list, std_method: str):
    """Allows to compute two level crossvalidation.

    Args:
        X (np.array): Features (numeric)
        y (np.array): Class (objective variable)
        k1 (int): Nº of outer folds
        k2 (int): Nº of inner folds
        models (list): List of models for comparison
        params (dict): Dictionary including the set of parameters. In this case we only tune 1 parameter per model.
        rs (int): Random state
    Returns:
        df: Dataframe
    """
    k = 0
    min_error = np.inf
    min_param = None
    names = [type(m).__name__ for m in models]
    col_names = ['Param. Value', 'Error']
    results_df = dataset_creator(fill_methods, names, col_names, k1)
    kf1 = KFold(k1, shuffle = True, random_state=rs)
    X_raw = df.drop('y', axis=1)
    y = df['y']
    # first level split
    for train_idx1, test_idx1 in kf1.split(X_raw, y):
        error_test = {}
        k += 1
        kf2 = KFold(k2, shuffle = True, random_state=rs)
        print(f'Computing KFold {k}/{k1}...')
        # second level split
        mean_errors = []
        for train_idx2, test_idx2 in tqdm(kf2.split(X_raw.iloc[train_idx1, :], y[train_idx1]), total = k2):
            for method in fill_methods:
                X_train = transform_data(X_raw.iloc[train_idx2, :], fill_method=method, std_method=std_method).values
                y_train = y[train_idx2]
                X_test = transform_data(X_raw, fill_method=method, std_method=std_method).iloc[test_idx2, :].values
                y_test = y[test_idx2]
                for name, model in zip(names, models):
                    grid = ParameterGrid(params[name])
                    for p_ in grid:
                        model = model.set_params(**p_)
                        # train the model
                        model.fit(X_train, y_train)
                        # evaluate performance
                        pred2_test = model.predict(X_test)
                        error = mse(pred2_test, y_test)
                        if error < min_error:
                            min_error = error
                            min_param = p_


    # results_df.loc(axis = 1)[method, name, 'Error'][k] = error_test[idx]
    # results_df.loc(axis = 1)[method, name, 'Param. Value'][k] = min_param
    return error_test, test_idx1

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self, X,y=None):
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
        return self.fit(X, y).transform(X)

def standarize(df,method):
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

