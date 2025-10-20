import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from src.prepare_data_for_model import prepare_data
from sklearn.model_selection import cross_val_score, StratifiedKFold


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