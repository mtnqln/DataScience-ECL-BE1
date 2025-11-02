import numpy as np
import pandas as pd
import json

def handle_labels_ids(labels:np.ndarray)->pd.Series:
    users = sorted(set(labels))
    user2id ={u:i for i,u in enumerate(users)}
    id2user = {i:u for i,u in user2id.items()}
    with open("src/models/transformer/results/id2user.json") as fp:
        json.dump(id2user,fp)
    with open("src/models/transformer/results/user2id.json") as fp:
        json.dump(user2id,fp)
    output = [user2id[label] for label in labels]
    return pd.Series(output)

def decode_labels_ids(id):
    with open("src/models/transformer/results/id2user.json") as rd:
        id2user:dict[int,str] = json.load(rd)
    output = id2user[id]
    return output
