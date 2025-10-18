import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import XGBClassifier


def xgboost_inference(X_train: np.ndarray, Y_train: np.ndarray, X_predict: np.ndarray) -> np.ndarray:
    """
    Entraîne un modèle XGBoost multiclasses à partir d'un dataset de sessions utilisateurs,
    puis prédit les utilisateurs correspondants pour X_predict.

    Returns
    -------
    np.ndarray
        Matrice (m_samples, n_classes) des prédictions, au format one-hot.
    """

    # --- Étape 1 : convertir Y_train en labels entiers ---
    if not isinstance(Y_train, np.ndarray):
        Y_train = Y_train.toarray()
    y_labels = np.argmax(Y_train, axis=1)

    # --- Étape 2 : entraîner le modèle ---
    num_classes = Y_train.shape[1]
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        verbosity=0
    )

    model.fit(X_train, y_labels)

    # --- Étape 3 : prédire les classes ---
    y_pred_labels = model.predict(X_predict)

    # --- Étape 4 : reconvertir en one-hot ---
    y_pred_onehot = np.zeros((len(y_pred_labels), num_classes))
    y_pred_onehot[np.arange(len(y_pred_labels)), y_pred_labels] = 1

    return y_pred_onehot
