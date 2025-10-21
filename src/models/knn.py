import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from src.prepare_data_for_model import prepare_data
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data_for_cross_val, prepare_data_for_prediction, prepare_data


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
    
    f1score = f1_score(y_true=y_true,y_pred=y_pred,average='macro')
    if isinstance(f1score,float):
        return f1score
    else:
        return 0.0 
    

def knn_cross_validation(X: np.ndarray, Y: np.ndarray, number_of_neighbors:int) -> float:
    """
    Effectue une validation croisée sur un modèle KNN.

    Parameters
    ----------
    X : np.ndarray
        Les caractéristiques d'entrée.
    Y : np.ndarray
        Les étiquettes cibles.
    number_of_neighbors : int
        Le nombre de voisins à utiliser dans le modèle KNN.

    Returns
    -------
    float
        La moyenne des scores de validation croisée.
    """
    model = KNeighborsClassifier(n_neighbors=number_of_neighbors)

    cv_folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    f1_score = cross_val_score(model, X, Y, cv=cv_folds, scoring='f1_macro')

    return f1_score.mean()


def knn_submission(number_of_neighbors:int)->np.ndarray:
    features_train = read_ds("data/train.csv")
    test = read_ds("data/test.csv")
    submission = pd.read_csv("data/sample_submission.csv")

    X_train, Y_train = prepare_data_for_cross_val(features_train)
    X_predict = prepare_data_for_prediction(test)
    # X_predict = align_features(X_train, X_predict)

    y_pred = knn_inference(X_train=X_train, Y_train=Y_train, X_predict=X_predict, number_of_neighbors=1)
    submission['prediction'] = y_pred
    submission.to_csv("data/xgboost_submission.csv", index=False)

    return submission