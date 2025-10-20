import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


def xgboost_inference(X_train: np.ndarray, Y_train: np.ndarray, X_predict: np.ndarray) -> np.ndarray:
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
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        verbosity=0
    )

    model.fit(X_train, y_encoded)

    # --- Étape 3 : prédire les classes encodées ---
    y_pred_encoded = model.predict(X_predict)

    # --- Étape 4 : reconvertir les entiers en labels utilisateur d'origine ---
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    return y_pred_labels