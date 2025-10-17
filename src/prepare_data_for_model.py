import pandas as pd
from .get_stat_from_data import (
    browsers_per_player,
    get_actions_frequency,
    get_mean_time,
)
from .helper import normalize_df
from sklearn.preprocessing import normalize,OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Any, Tuple


### Prepare Data ### 

def prepare_data(df:pd.DataFrame) -> Tuple[Any, ...]:
    """Take the input data and return a train and test dataset"""
    mean_time = normalize_df(get_mean_time(df=df),not_cols=0)
    print("Mean time ",mean_time.iloc[0,:],"\n")
    browsers_p_player = normalize_df(browsers_per_player(df=df,normalize=True,unique=False),not_cols=0)
    print("Browser p player",browsers_p_player.iloc[0,:],"\n")
    actions_frequency = normalize_df(get_actions_frequency(df=df),not_cols=0)
    print("\n Action frequency : ",actions_frequency.iloc[0,:])

    df_buff = pd.merge(actions_frequency,browsers_p_player,on=0)
    df_training = pd.merge(df_buff,mean_time,on=0)

    # Getting labels
    y = df_training[0]
    print("\n Xtrain : ",df_training.iloc[:,:])
    X = df_training.drop(0,axis=1)

    #OneHotEncoding
    encoder = OneHotEncoder(handle_unknown='ignore')
    y = y.to_numpy().reshape(-1,1)
    encoded_y = encoder.fit_transform(y)
    X_train,X_test,y_train,y_test = train_test_split(X,encoded_y,test_size=0.10)
    print("Xtrain : ",X_train)
    print("Xtest : ",X_test)
    return X_train,X_test,y_train,y_test




### Prepare Data for KNN ### 
# def prepare_data_for_knn(df:pd.DataFrame)->pd.DataFrame:

#     pass


