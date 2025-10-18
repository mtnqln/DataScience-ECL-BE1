import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from src.prepare_data_for_model import prepare_data


def knn_inference(X_train:np.ndarray,Y_train:np.ndarray,X_predict:np.ndarray,number_of_neighbors:int)->np.ndarray:
    """Takes the input datasets and return the prediction vector"""
    # Training O(1)
    neigh = KNeighborsClassifier(n_neighbors=number_of_neighbors)
    neigh.fit(X=X_train,y=Y_train) #type: ignore

    # Inference O(n)
    result = neigh.predict(X_predict)
    return result

def knn_f1_score(features_train:pd.DataFrame,nn:int)->float:
    """Take the input dataset, return the f1 score on the dataset"""
    X_train,X_test,y_train,y_true = prepare_data(features_train)
    y_pred = knn_inference(X_train=X_train,Y_train=y_train,X_predict=X_test,number_of_neighbors=nn)
    print("\n y pred : ",y_pred)
    print("\n y true : ",y_true)
    f1score = f1_score(y_true=y_true,y_pred=y_pred,average='macro')
    if isinstance(f1score,float):
        return f1score
    else:
        return 0.0 