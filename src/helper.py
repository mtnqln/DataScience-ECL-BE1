import pandas as pd


def normalize_df(df:pd.DataFrame,not_cols:int)->pd.DataFrame:
    """Normalize a df row by row ignoring the first not_cols columns"""
    prefix_df = df.iloc[:,:not_cols+1]
    to_normalize_df = df.iloc[:,not_cols+1:]
    rows_sum = to_normalize_df.sum(axis=1).replace(0,1)
    normalized = to_normalize_df[:].div(rows_sum,axis=0)

    new_df = pd.concat([prefix_df,normalized],axis=1)
    return new_df