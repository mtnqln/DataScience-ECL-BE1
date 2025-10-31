from src.models.transformer.tokenizer import create_and_train_tokenizer
import os 

def test_tokenizer():
    create_and_train_tokenizer()
    file_path = "src/models/transformer/results/tokenizer.json"
    assert os.path.exists(file_path)

def test_result_tokenizer():
    pass