import numpy as np
import pandas as pd


def handle_labels_ids(labels:np.ndarray)->pd.Series:
    users = sorted(set(labels))
    user2id ={u:i for i,u in enumerate(users)}
    id2user = {i:u for i,u in user2id.items()}
    output = [user2id[label] for label in labels]
    return pd.Series(output)
