import numpy as np
from sklearn.metrics import f1_score
from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data
from src.prepare_data_for_model import prepare_data_for_xgboost
from src.models.xgboost import xgboost_inference
from tqdm import tqdm


def main():
    print("Hello from datascience-ecl-be1!")


if __name__ == "__main__":
    features_train = read_ds("data/train.csv")
    breakpoint()
    X_train,X_test,y_train,y_test = prepare_data_for_xgboost(features_train)
    BATCH_SIZE = 5000
    all_preds = []
    all_true = []

    n = len(X_test)

    for start in tqdm(range(0, n, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, n)
        X_batch = X_test[start:end]
        y_batch = y_test[start:end]

        # 🔮 prédiction du batch
        y_pred_batch = xgboost_inference(X_train=X_train, Y_train=y_train, X_predict=X_batch)
        
        # 🧩 on stocke les résultats
        all_preds.append(y_pred_batch)
        all_true.append(y_batch)

    # on rassemble tout à la fin
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    #  score global
    f1 = f1_score(all_true, all_preds, average='weighted')
    print(f"F1-score global (batché) : {f1:.4f}")
