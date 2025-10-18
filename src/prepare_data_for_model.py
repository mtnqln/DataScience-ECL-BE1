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
    #print("\n Xtrain : ",df_training.iloc[:,:])
    X = df_training.drop(0,axis=1)
    print("#############################")
    print(y)
    print("#############################")
    #OneHotEncoding
    encoder = OneHotEncoder(handle_unknown='ignore')
    y = y.to_numpy().reshape(-1,1)
    encoded_y = encoder.fit_transform(y)
    X_train,X_test,y_train,y_test = train_test_split(X,encoded_y,test_size=0.10)
    #print("Xtrain : ",X_train)
    #print("Xtest : ",X_test)
    return X_train,X_test,y_train,y_test


def prepare_data_for_xgboost(df:pd.DataFrame) -> Tuple[Any, ...]:
    """Take the input data and return a train and test dataset"""
    mean_time = normalize_df(get_mean_time(df=df),not_cols=0)
    #print("Mean time ",mean_time.iloc[0,:],"\n")
    browsers_p_player = normalize_df(browsers_per_player(df=df,normalize=True,unique=False),not_cols=0)
    #print("Browser p player",browsers_p_player.iloc[0,:],"\n")
    actions_frequency = normalize_df(get_actions_frequency(df=df),not_cols=0)
    #print("\n Action frequency : ",actions_frequency.iloc[0,:])

    df_training = merge_in_batches(actions_frequency, browsers_p_player, mean_time, batch_size=5000)

    # Getting labels
    y = df_training[0]
    #print("\n Xtrain : ",df_training.iloc[:,:])
    X = df_training.drop(0,axis=1)
    print("#############################")
    print(y)
    print("#############################")
    y = y.to_numpy().reshape(-1,1)[:,0]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10)
    #print("Xtrain : ",X_train)
    #print("Xtest : ",X_test)
    return X_train,X_test,y_train,y_test

def merge_in_batches(actions_frequency, browsers_p_player, mean_time, batch_size=10_000):
    """
    Fusionne trois DataFrames par petits lots pour éviter la surcharge mémoire.
    On suppose que tous ont une colonne commune appelée '0'.
    """

    all_keys = actions_frequency[0].unique()
    n = len(all_keys)
    print(f"[INFO] Nombre total d'IDs à fusionner : {n:,}")

    results = []  # pour stocker les morceaux fusionnés

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        keys_batch = all_keys[start:end]

        a_batch = actions_frequency[actions_frequency[0].isin(keys_batch)]
        b_batch = browsers_p_player[browsers_p_player[0].isin(keys_batch)]
        m_batch = mean_time[mean_time[0].isin(keys_batch)]

        # 🔸 On fait les merges sur le batch courant
        df_buff = pd.merge(a_batch, b_batch, on=0, how="inner")
        df_training_batch = pd.merge(df_buff, m_batch, on=0, how="inner")

        results.append(df_training_batch)

        print(f"[INFO] Batch {start//batch_size + 1} fusionné ({len(df_training_batch):,} lignes)")

    df_training = pd.concat(results, ignore_index=True)
    print(f"[INFO] Fusion complète terminée : {len(df_training):,} lignes")

    return df_training


### Prepare Data for KNN ### 
# def prepare_data_for_knn(df:pd.DataFrame)->pd.DataFrame:

#     pass
