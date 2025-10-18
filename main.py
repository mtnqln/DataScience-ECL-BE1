import numpy as np

from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data

def main():
    print("Hello from datascience-ecl-be1!")


if __name__ == "__main__":
    features_train = read_ds("data/train.csv").loc[:100,]
    X_train,X_test,y_train,y_test = prepare_data(features_train) 
