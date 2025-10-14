import pandas as pd
import warnings
from IPython.display import display, Markdown

file_path = "C:/Users/ledou/OneDrive/Bureau/BE_datascience/test_compresse.csv.gz"

# décorateurs utilitaires pour supprimer les avertissements de la sortie et imprimer un cadre de données dans un tableau Markdown.
def ignore_warnings(f):
    def _f(*args, **kwargs):
        warnings.filterwarnings('ignore')
        v = f(*args, **kwargs)
        warnings.filterwarnings('default')
        return v
    return _f

column_names = [f"col_{i}" for i in range(14470)]
column_names[0]='util'
column_names[1]='navigateur'

file = pd.read_csv(file_path, names=column_names)
print(file.head())