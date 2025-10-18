import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.handle_data_pandas import read_ds
from src.models.xgboost import xgboost_inference
from src.prepare_data_for_model import prepare_data, prepare_data_for_xgboost
def main():
    features_train = read_ds("data/train.csv")
    X_train,X_test,y_train,y_test = prepare_data(features_train) 
    print(X_train.shape)

if __name__ == "__main__":
    features_train = read_ds("data/train.csv")
    X_train, X_test, y_train, y_test = prepare_data_for_xgboost(features_train)

    y_pred = xgboost_inference(X_train=X_train, Y_train=y_train, X_predict=X_test)

    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' pour gérer les classes déséquilibrées
    print(f"F1-score global : {f1:.4f}")
