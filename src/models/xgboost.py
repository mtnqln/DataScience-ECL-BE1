import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold , GridSearchCV

from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data_for_prediction


def xgboost_inference(X_train: np.ndarray, 
                      Y_train: np.ndarray, 
                      X_predict: np.ndarray) -> np.ndarray:
    """
    Entraîne un modèle XGBoost multiclasses avec recherche de grille (GridSearchCV),
    puis prédit les utilisateurs correspondants pour X_predict.

    Returns
    -------
    np.ndarray
        Tableau 1D des prédictions (m_labels) correspondant aux utilisateurs.
    """

    # --- Étape 1 : encoder les labels texte en entiers ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(Y_train)

    # --- Étape 2 : définir le modèle de base ---
    num_classes = len(le.classes_)
    base_model = XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        verbosity=0
    )

    # --- Étape 3 : définir la grille de recherche ---
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1]
    }

    # --- Étape 4 : définir la cross-validation ---
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # --- Étape 5 : lancer la recherche de grille ---
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_encoded)

    # --- Étape 6 : utiliser le meilleur modèle pour la prédiction ---
    best_model = grid.best_estimator_
    y_pred_encoded = best_model.predict(X_predict)

    # --- Étape 7 : reconvertir les entiers en labels utilisateur d'origine ---
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    return y_pred_labels




def xgboost_cross_validation(X: np.ndarray, 
                             Y: np.ndarray) -> dict:
    """
    Effectue une validation croisée avec recherche de grille
    sur un modèle XGBoost multiclasses.

    Parameters
    ----------
    X : np.ndarray
        Caractéristiques d'entrée.
    Y : np.ndarray
        Étiquettes cibles.

    Returns
    -------
    dict
        Résumé du meilleur modèle : hyperparamètres, score moyen.
    """

    # --- Étape 1 : encoder les labels ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(Y)

    # --- Étape 2 : définition du modèle ---
    model = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        tree_method="hist",
        verbosity=0
    )

    # --- Étape 3 : grille de recherche ---
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1]
    }

    # --- Étape 4 : validation croisée stratifiée ---
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # --- Étape 5 : recherche par grille ---
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X, y_encoded)

    # --- Étape 6 : résumé du meilleur modèle ---
    return {
        "best_params": grid.best_params_,
        "best_score": grid.best_score_
    }

def xgboost_submit():
    features_train = read_ds("data/train.csv")
    features_test = read_ds("data/test.csv", test=True)
    submission = pd.read_csv("data/sample_submission.csv")

    X_train, Y_train, X_predict = prepare_data_for_prediction(features_train, features_test)

    y_pred = xgboost_inference(X_train=X_train, 
                               Y_train=Y_train, 
                               X_predict=X_predict)
    
    submission['prediction'] = y_pred
    submission.to_csv("submission/xgboost_submission.csv", index=False)