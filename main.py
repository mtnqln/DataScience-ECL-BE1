import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd

from src.handle_data_pandas import read_ds
from src.models.xgboost import xgboost_inference, xgboost_cross_validation
from src.models.knn import knn_inference, knn_f1_score, knn_cross_validation
from src.prepare_data_for_model import prepare_data, prepare_data_for_cross_val, prepare_data_for_prediction, align_features

def main():
    pass
if __name__ == "__main__":
    features_train = read_ds("data/train.csv")
    test = read_ds("data/test.csv")
    submission = pd.read_csv("data/sample_submission.csv")
    # X_train, X_test, y_train, y_test = prepare_data_for_xgboost(features_train)

    # y_pred = xgboost_inference(X_train=X_train, Y_train=y_train, X_predict=X_test)

    # f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' pour gérer les classes déséquilibrées

    # print(f"F1-score global : {f1:.4f}")


    X, Y = prepare_data_for_cross_val(features_train)
    X_test = prepare_data_for_prediction(test)
    X_test = align_features(X, X_test)
    y_pred = xgboost_inference(X_train=X, Y_train=Y, X_predict=X_test)
    print(y_pred)
    submission['prediction'] = y_pred
    submission.to_csv("data/xgboost_submission.csv", index=False)

    f1_score = xgboost_cross_validation(X, Y)
    print(f"Cross-validation scores: {f1_score}")


    # X, Y = prepare_data_for_cross_val(features_train)
    # f1_score = knn_cross_validation(X, Y, 1)
    # print(f"Cross-validation scores: {f1_score}")