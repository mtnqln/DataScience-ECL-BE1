from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from src.handle_data_pandas import read_ds
from tokenizers import decoders

### Training
def get_training_corpus():
    df_train = read_ds("data/train.csv")
    df_train = df_train.fillna("").drop(0,axis=1)
    df_train["trace"] = df_train.apply(lambda row: ",".join([str(x) for x in row if x != ""]), axis=1)

    for text in df_train["trace"]:
        yield text


def create_and_train_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # type: ignore
    trainer = trainers.BpeTrainer(vocab_size=27000, special_tokens=["<|endoftext|>","<PAD>","<UNK>"])
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel() # type: ignore
    tokenizer.save("src/models/transformer/results/tokenizer.json")

