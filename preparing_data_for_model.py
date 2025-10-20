import get_stat_from_data
import pandas as pd

def data_for_model(list):
    return pd.concat(list, axis=1)

def OHE(df):
    ohe_df = pd.get_dummies(df)
    return ohe_df

