from src.models.knn import submit_knn
from src.models.xgboost import xgboost_submit, xgboost_cross_validation
from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data_for_prediction
import pandas as pd

def main():
    # submit_knn()
    # xgboost_submit()
    features_train = read_ds("data/train.csv")
    features_test = read_ds("data/test.csv", test=True)
    submission = pd.read_csv("data/sample_submission.csv")

    X_train, Y_train, X_predict = prepare_data_for_prediction(features_train, features_test)
    y_pred = xgboost_cross_validation(X_train=X_train, y_train=Y_train)


if __name__ == "__main__":
    main()
    
