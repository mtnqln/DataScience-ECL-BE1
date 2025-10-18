import numpy as np
from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data

def main():
    features_train = read_ds("data/train.csv")
    X_train,X_test,y_train,y_test = prepare_data(features_train) 
    print(X_train.shape)

if __name__ == "__main__":
    main()
    