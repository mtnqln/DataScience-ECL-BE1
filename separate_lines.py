import re
from collections import Counter
from handle_data_pandas import read_ds
import pandas as pd





if __name__ == "__main__":

    features_train = read_ds("data/train.csv")
    features_test = read_ds("data/test.csv")

    pattern_ecran = re.compile(r"\((.*?)\)")
    pattern_conf_ecran = re.compile(r"<(.*?)>")
    pattern_chaine = re.compile(r"\$(.*?)\$")
    
    features_train["util"] = pd.Categorical(features_train["util"])

    def two_pattern(df):
        two_pattern = []
        for _, row in df.iterrows():
            actions = row.dropna().tolist()  # retire les NaN au cas où
            for i in range(len(actions) - 1):
                two_pattern.append((actions[i], actions[i + 1]))