from src.models.xgboost import xgboost_inference
from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data_for_xgboost
from sklearn.metrics import f1_score
import numpy as np

#rajoute les cycles batards

def test_xgboost_inference():
    features_train = read_ds("data/train.csv")
    X_train, X_test, y_train, y_test = prepare_data_for_xgboost(features_train)

    y_pred = xgboost_inference(X_train=X_train, Y_train=y_train, X_test=X_test)
    print(np.unique(y_test[:10]), y_test.dtype)
    f1 = f1_score(y_test.astype(int), y_pred.astype(int), average='weighted')  
    print(f"F1-score global : {f1:.4f}")
    assert f1 > 0.7
