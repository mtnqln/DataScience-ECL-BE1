import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def knn_inference(X_train:np.ndarray,Y_train:np.ndarray,X_predict:np.ndarray,number_of_neighbors:int)->np.ndarray:
    # Training O(1)
    neigh = KNeighborsClassifier(n_neighbors=number_of_neighbors)
    neigh.fit(X=X_train,y=Y_train) #type: ignore

    # Inference O(n)
    result = neigh.predict(X_predict)
    return result
