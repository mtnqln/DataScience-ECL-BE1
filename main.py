from src.models.knn import submit_knn
from src.models.xgboost import xgboost_submit

def main():
    submit_knn()
    xgboost_submit()

if __name__ == "__main__":
    main()
    
