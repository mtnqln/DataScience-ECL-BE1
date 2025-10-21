import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data_for_prediction

def knn_inference(X_train:np.ndarray,Y_train:np.ndarray,X_predict:np.ndarray,number_of_neighbors:int)->np.ndarray:
    """Takes the input datasets and return the prediction vector"""
    # Training O(1)
    neigh = KNeighborsClassifier(n_neighbors=number_of_neighbors)
    neigh.fit(X=X_train,y=Y_train) #type: ignore

    # Inference O(n)
    result = neigh.predict(X_predict)
    return result

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


def submit_knn():
    features_train = read_ds("data/train.csv")
    features_test = read_ds("data/test.csv",test=True)

    X_train, Y_train, X_predict = prepare_data_for_prediction(features_train, features_test)

    y_pred = knn_inference(X_train=X_train, Y_train=Y_train, X_predict=X_predict, number_of_neighbors=1)

    submission = pd.DataFrame()
    submission["RowId"] = pd.Series(list(range(1,len(y_pred)+1)))
    submission['prediction'] = y_pred
    
    submission.to_csv("submission/submission_sample_knn.csv", index=False)
