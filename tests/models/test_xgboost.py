from src.models.xgboost import xgboost_inference
from src.helper import score_accuracy
from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data_for_xgboost
from sklearn.metrics import f1_score
import numpy as np

#rajoute les cycles batards

def test_xgboost_inference():
    features_train = read_ds("data/train.csv").loc[:100,]
    X_train,X_test,y_train,y_test = prepare_data_for_xgboost(features_train)
    y_pred = xgboost_inference(X_train=X_train,Y_train=y_train,X_predict=X_test)
    print("###############predit##############################")
    print(y_pred)
    print("###########predit#######################")
    print("###########reel###############")
    print(y_test)
    print("#############reel###########")
    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' gère bien les classes déséquilibrées
    print(f'************************ F1-score: {f1:.4f} **********************')
    #accuracy = score_accuracy(y_test,y_pred)
    #print("Accuracy : ",accuracy)
    #assert accuracy > 0.8
