import pandas as pd

### Reading data from csv ###
def read_ds(ds_name:str,
            test=False)->pd.DataFrame:
    with open(ds_name,"r",encoding="utf-8") as f:
        max_cols = max(len(line.strip().split(",")) for line in f)

    if test:
        col_names = list(range(1, max_cols+1))
    else:
        col_names = list(range(max_cols))

    df = pd.read_csv(
        ds_name,
        header=None,
        names=col_names,
        engine="python"
    )
    if test:
        df.insert(0, 0, "a")
    return df

### Showing Data samples ###
if __name__=="__main__":
    features_train = read_ds("data/train.csv")
    features_test = read_ds("data/test.csv")
    print("Train head : \n",features_train.head())
    print("Test : \n ",features_train.loc[:,:20].head())