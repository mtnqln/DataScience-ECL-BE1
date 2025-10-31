from src.models.knn import submit_knn
# from src.models.xgboost import xgboost_submit
from src.models.transformer.train_transformer import train_transformer
 
def main():
    # submit_knn()
    # xgboost_submit()
    train_transformer()

if __name__ == "__main__":
    main()
    