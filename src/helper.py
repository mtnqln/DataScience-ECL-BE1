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

