import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.metrics import accuracy_score


def normalize_df(df:pd.DataFrame,not_cols:int)->pd.DataFrame:
    """Normalize a df row by row ignoring the first not_cols columns"""
    prefix_df = df.iloc[:,:not_cols+1]
    to_normalize_df = df.iloc[:,not_cols+1:]
    rows_sum = to_normalize_df.sum(axis=1).replace(0,1)
    normalized = to_normalize_df[:].div(rows_sum,axis=0)

    new_df = pd.concat([prefix_df,normalized],axis=1)
    return new_df

def last_non_na_from_tuple(row_tuple):
    # if your tuple includes the index in position 0, skip it: row_tuple = row_tuple[1:]
    for v in reversed(row_tuple):
        if pd.notna(v):           # treats both None and np.nan as NA
            return v
    return np.nan


#TODO test those functions
def to_label_1d(y):
    """Return a 1-D array of class labels from various common forms."""
    # 1) Sparse one-hot
    if sp.issparse(y):
        return y.argmax(axis=1).A1

    # 2) Pandas containers
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    elif isinstance(y, pd.DataFrame):
        arr = y.to_numpy()
        # If it looks like one-hot, argmax across columns
        if arr.ndim == 2 and arr.shape[1] > 1 and np.isin(arr, [0, 1]).all():
            return arr.argmax(axis=1)
        # Else, take the first column
        return arr[:, 0].ravel()

    # 3) NumPy / Python containers
    y = np.asarray(y, dtype=object)

    # If it's a 2-D numeric array (likely one-hot/probs): argmax on axis=1
    if y.ndim == 2 and np.issubdtype(np.array(y).dtype, np.number):
        return np.asarray(y, dtype=float).argmax(axis=1)

    # If it's 1-D but each element is an array/list (prob vectors)
    if y.ndim == 1 and len(y) > 0 and isinstance(y[0], (list, np.ndarray)):
        stacked = np.vstack([np.asarray(row) for row in y])
        return stacked.argmax(axis=1)

    # Otherwise assume it's already 1-D labels
    return y.ravel()

def score_accuracy(y_true, y_pred):
    yt = to_label_1d(y_true)
    yp = to_label_1d(y_pred)
    return accuracy_score(yt, yp)