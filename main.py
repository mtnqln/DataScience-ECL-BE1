import numpy as np
import get_stat_from_data

def main():
    pass
if __name__ == "__main__":
    features_train = read_ds("data/train.csv")
    X_train, X_test, y_train, y_test = prepare_data_for_xgboost(features_train)

    y_pred = xgboost_inference(X_train=X_train, Y_train=y_train, X_predict=X_test)

    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' pour gérer les classes déséquilibrées
    print(f"F1-score global : {f1:.4f}")
