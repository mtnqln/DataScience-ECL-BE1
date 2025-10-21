import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data_for_prediction


def xgboost_inference(X_train: np.ndarray, 
                      Y_train: np.ndarray, 
                      X_predict: np.ndarray,
                      n_estimators=200,
                      max_depth=6,) -> np.ndarray:
    """
    Entraîne un modèle XGBoost multiclasses à partir d'un dataset de sessions utilisateurs,
    puis prédit les utilisateurs correspondants pour X_predict.

    Returns
    -------
    np.ndarray
        Tableau 1D des prédictions (m_labels) correspondant aux utilisateurs.
    """

    # --- Étape 1 : encoder les labels texte en entiers ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(Y_train)

    # --- Étape 2 : entraîner le modèle ---
    num_classes = len(le.classes_)
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        verbosity=0
    )

    model.fit(X_train, y_encoded)

    # --- Étape 3 : prédire les classes encodées ---
    y_pred_encoded = model.predict(X_predict)

    # --- Étape 4 : reconvertir les entiers en labels utilisateur d'origine ---
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    return y_pred_labels



def xgboost_cross_validation(X: np.ndarray, 
                             Y: np.ndarray, 
                             n_estimators=200, 
                             max_depth=6) -> float:
    """
    Effectue une validation croisée sur un modèle XGBoost régressif.

    Parameters
    ----------
    X : np.ndarray
        Les caractéristiques d'entrée.
    Y : np.ndarray
        Les étiquettes cibles.

    Returns
    -------
    float
        La moyenne des scores de validation croisée.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(Y)

    # --- Étape 2 : entraîner le modèle ---
    num_classes = len(le.classes_)
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        verbosity=0
    )

    cv_folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    f1_score = cross_val_score(model, X, y_encoded, cv=cv_folds, scoring='f1_macro')

    return f1_score.mean()

def xgboost_submit():
    features_train = read_ds("data/train.csv")
    features_test = read_ds("data/test.csv", test=True)
    submission = pd.read_csv("data/sample_submission.csv")

    X_train, Y_train, X_predict = prepare_data_for_prediction(features_train, features_test)

    y_pred = xgboost_inference(X_train=X_train, 
                               Y_train=Y_train, 
                               X_predict=X_predict, 
                               n_estimators=300, 
                               max_depth=6)
    
    submission['prediction'] = y_pred
    submission.to_csv("submission/xgboost_submission.csv", index=False)