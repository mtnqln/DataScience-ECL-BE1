from src.helper import score_accuracy
from src.models.knn import knn_inference
from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data

def test_knn_inference():
    features_train = read_ds("data/train.csv").loc[:100,]
    X_train,X_test,y_train,y_test = prepare_data(features_train)
    y_pred = knn_inference(X_train=X_train,Y_train=y_train,X_predict=X_test,number_of_neighbors=8)

    accuracy = score_accuracy(y_test,y_pred)
    print("Accuracy : ",accuracy)
    assert accuracy > 0.8




