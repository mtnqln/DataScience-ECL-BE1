import pandas as pd
from .get_stat_from_data import (
    browsers_per_player,
    get_actions_frequency,
    get_mean_time,
    get_normalize_browser_distribution,
)
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

### Prepare Data ### 
from typing import Any, Tuple

def prepare_data(df:pd.DataFrame) -> Tuple[Any, ...]:
    """Take the input data and return a train, validation and test dataset"""
    normalized_browsers = get_normalize_browser_distribution(df=df)
    mean_time = get_mean_time(df=df)
    browsers_p_player = browsers_per_player(df=df,normalize=True)
    actions_frequency = get_actions_frequency(df=df)

    df_training = pd.DataFrame()
    df_training["target"] = df.loc[:,0]
    df_training.set_index("target")
    df_buff = pd.concat([browsers_p_player, mean_time,actions_frequency], axis=1)
    df_buff = pd.DataFrame(normalize(df_buff),columns=df_buff.columns,index=df_buff.index)
    df_training.join(df_buff,on="index")

    X = df_training.drop("target")
    y = df_training["target"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10)
    
    return X_train,X_test,y_train,y_test





### Prepare Data for KNN ### 
def prepare_data_for_knn(df:pd.DataFrame)->pd.DataFrame:

    return



### Function tests
if __name__=="__main__":
    pass
