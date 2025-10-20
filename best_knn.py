import numpy as np
from src.handle_data_pandas import read_ds
from src.models.knn import knn_f1_score
from src.prepare_data_for_model import prepare_data
import matplotlib.pyplot as plt

from src.models.knn import knn_cross_validation
from src.prepare_data_for_model import prepare_data_for_cross_val


def best_knn():
    features_train = read_ds("data/train.csv")
    scores:list[float] = []
    for k in range(1,15):
        # score = knn_f1_score(features_train=features_train,nn=k)
        X, Y = prepare_data_for_cross_val(features_train)
        score = knn_cross_validation(X, Y, k)

        scores.append(score)
        print(f"F1 score for {k} neighbors: {score}")
    fig, ax = plt.subplots()     
    ax.plot(range(1,15), scores)  
    plt.show()  
    return (scores.index(max(scores))+1,max(scores))
    
    

        
if __name__ == "__main__":
    best_value = best_knn()
    print(f"Best possible knn is : {best_value}")