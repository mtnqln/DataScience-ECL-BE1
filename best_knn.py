import numpy as np
from src.handle_data_pandas import read_ds
import matplotlib.pyplot as plt

from src.models.knn import knn_cross_validation
from src.models.xgboost import xgboost_cross_validation
from src.prepare_data_for_model import prepare_data_for_prediction, prepare_data_for_cross_val


# def best_knn():
#     features_train = read_ds("data/train.csv")
#     scores:list[float] = []
#     for k in range(1,15):
#         # score = knn_f1_score(features_train=features_train,nn=k)
#         X_train, Y_train, X_predict = prepare_data_for_prediction(features_train, test)
#         score = knn_cross_validation(X_train, Y_train, k)

#         scores.append(score)
#         print(f"F1 score for {k} neighbors: {score}")
#     fig, ax = plt.subplots()     
#     ax.plot(range(1,15), scores)  
#     plt.show()  
#     return (scores.index(max(scores))+1,max(scores))
    
    
def best_xgboost():
    features_train = read_ds("data/train.csv")
    test = read_ds("data/test.csv", test=True)
    scores:list[float] = []
    for k in [4, 6, 8]:
        # score = knn_f1_score(features_train=features_train,nn=k)
        X_train, Y_train, X_predict = prepare_data_for_prediction(features_train, test)
        score = xgboost_cross_validation(X_train, Y_train, max_depth=k, n_estimators=300)
        print(f"F1 score for {k} estimators: {score}")

    #     scores.append(score)
    #     print(f"F1 score for {k} neighbors: {score}")
    # fig, ax = plt.subplots()     
    # ax.plot(range(1,15), scores)  
    # plt.show()  
    # return (scores.index(max(scores))+1,max(scores))
        
if __name__ == "__main__":
    best_value = best_xgboost()
    print(f"Best possible xgboost is : {best_value}")