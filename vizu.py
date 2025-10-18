import pandas as pd

n_cols = 14470

col_names = [f"col {i}" for i in range(n_cols)]

features_train = pd.read_csv("data/train.csv.GZ",
                            header=None,
                            names=col_names,
                            engine="python"
                            )
features_test = pd.read_csv("data/test.csv.GZ",
                            header=None,
                            names=col_names,
                            engine="python"
                            )
print("Shape : ",features_train.shape,features_test.shape)

print("Train head : ",features_train.head())