import pandas as pd
from .get_stat_from_data import (
    browsers_per_player,
    get_actions_frequency,
    get_mean_time,
)
from .helper import normalize_df
from sklearn.preprocessing import normalize,LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Any, Tuple


### Prepare Data ### 

def prepare_data(df:pd.DataFrame) -> Tuple[Any, ...]:
    """Take the input data and return a train and test dataset"""
    mean_time = normalize_df(get_mean_time(df=df),not_cols=0).drop(0,axis=1)
    browsers_p_player = normalize_df(browsers_per_player(df=df,normalize=True,unique=False),not_cols=0).drop(0,axis=1)
    actions_frequency = normalize_df(get_actions_frequency(df=df),not_cols=0)
    # For labels
    y = actions_frequency[0]
    actions_frequency = actions_frequency.drop(0,axis=1)
    df_buff = pd.merge(actions_frequency,browsers_p_player,left_index=True,right_index=True)
    df_training = pd.merge(df_buff,mean_time,left_index=True,right_index=True)


    #OneHotEncoding
    lb = LabelEncoder()
    y = y.to_numpy()
    lb.fit(y)
    encoded_y = lb.transform(y) # y is encoded as for example [ 0, 2, 1] where each number is a class
    X_train,X_test,y_train,y_test = train_test_split(df_training,encoded_y,test_size=0.20)
    return X_train,X_test,y_train,y_test




### Prepare Data for KNN ### 
# def prepare_data_for_knn(df:pd.DataFrame)->pd.DataFrame:

#     pass


