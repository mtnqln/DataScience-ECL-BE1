import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


def xgboost_inference(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Entraîne un modèle XGBoost multiclasses à partir d'un dataset de sessions utilisateurs,
    puis prédit les utilisateurs correspondants pour X_predict.

    Returns
    -------
    np.ndarray
        Tableau 1D des prédictions (m_labels) correspondant aux utilisateurs.
    """

    num_classes = len(Y_train)
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

    model.fit(X_train, Y_train)

    y_pred_encoded = model.predict(X_test)

    return y_pred_encoded