import pandas as pd
from tokenizers import Tokenizer
from torch.utils.data import DataLoader,Dataset
import torch
from src.handle_data_pandas import read_ds
from src.models.transformer.transformers_encoder import ClassifierEncoder
from src.models.transformer.utils import handle_labels_ids

class ClassificationDataset(Dataset):
    def __init__(self, labels:pd.Series, data:pd.Series, max_len:int,tokenizer:Tokenizer | None =None):
        self.labels = labels
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]

        if self.tokenizer:
            encoding = self.tokenizer.encode(text)
            ids = encoding.ids[:self.max_len]
            pad_id = self.tokenizer.token_to_id("<PAD>")
            ids = ids + [pad_id] * (self.max_len - len(ids))
            text = torch.tensor(ids)

        label = torch.tensor(label)

        return {
            "src": text,
            "tgt": label
        }


def prepare_data_for_transformer()->tuple[DataLoader,dict[str,int]]:
    # Data
    df_train = read_ds("data/train.csv")
    target = handle_labels_ids(df_train[0].to_numpy())
    df_train = df_train.fillna("").drop(0,axis=1)
    df_train["trace"] = df_train.apply(lambda row: ",".join([str(x) for x in row if x != ""]), axis=1)
    tokenizer:Tokenizer = Tokenizer.from_file("src/models/transformer/results/tokenizer.json")
    data = df_train["trace"]

    # Params
    params = {}
    max_seq_length = 0
    for element in df_train["trace"]:
        if len(element)>max_seq_length:
            max_seq_length = len(element)
    params["max_seq_length"] = max_seq_length
    tgt_vocab_size = target.nunique(dropna=True)
    src_vocab_size = tokenizer.get_vocab_size()
    print(f"TGT VCB SIZE : {tgt_vocab_size} \n")
    print(f"SRC VCB SIZE : {src_vocab_size} \n")
    params["tgt_vocab_size"] = tgt_vocab_size
    params["src_vocab_size"] = src_vocab_size

    # Overiding max seq len
    max_seq_length = 250
    print(f"\n MAX SEQ LEN : {max_seq_length} \n")
    train_dataset = ClassificationDataset(labels=target,data=data,max_len=max_seq_length,tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
    batch = next(iter(train_dataloader))
    print(type(batch["src"]), batch["src"].shape)
    print(type(batch["tgt"]), batch["tgt"].shape)
    print(f"Data example : input : {batch["src"][0]}, and label : {batch["tgt"][0]} \n")
    # test_dataset = ClassificationDataset()
    # test_dataloader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)

    print("Done preparing data for transformer ! \n")
    return train_dataloader,params

def prepare_transformer(params:dict[str,int])->ClassifierEncoder:

    src_vocab_size = params.get("src_vocab_size",5000)
    tgt_vocab_size = params.get("tgt_vocab_size",5000)
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = params.get("max_seq_length",200)
    dropout = 0.1

    transformer = ClassifierEncoder(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    
    print("Done preparing transformer ! \n")
    return transformer
