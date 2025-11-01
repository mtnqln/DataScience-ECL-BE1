import pandas as pd
from .get_stat_from_data import (
    browsers_per_player,
    get_actions_frequency,
    get_mean_time,
    get_consecutive_action_tuples
)
from .helper import normalize_df
from sklearn.preprocessing import normalize,LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Any, Tuple


### Prepare Data ### 

def prepare_data_for_prediction(df_train:pd.DataFrame, df_test:pd.DataFrame|None=None) -> Tuple[Any, ...]:
    """Take the input data and return a train and test dataset"""
    if isinstance(df_test,pd.DataFrame):
        df = pd.concat([df_train, df_test],ignore_index=True)

    mean_time = normalize_df(get_mean_time(df=df),not_cols=0).drop(0,axis=1)
    browsers_p_player = normalize_df(browsers_per_player(df=df,normalize=True,unique=False),not_cols=0).drop(0,axis=1)
    actions_frequency = normalize_df(get_actions_frequency(df=df),not_cols=0)
    tuples_consecutives_action = normalize_df(get_consecutive_action_tuples(df=df),not_cols=0).drop(0,axis=1)

    df_buff0 = pd.merge(actions_frequency,browsers_p_player,left_index=True,right_index=True)
    df_buff = pd.merge(df_buff0,tuples_consecutives_action,left_index=True,right_index=True)
    df_processed = pd.merge(df_buff,mean_time,left_index=True,right_index=True)
    
    
    Y_train = df_processed[df_processed[0]!="a"][0]
    X_train = df_processed[df_processed[0]!="a"].drop(0,axis=1)
    try:
        X_test = df_processed[df_processed[0]=="a"].drop(0,axis=1)
    except:
        X_test = None
        pass
    return X_train, Y_train, X_test
