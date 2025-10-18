import pandas as pd
from sklearn.metrics import f1_score
from src.models.knn import knn_inference, knn_f1_score
from src.handle_data_pandas import read_ds
from src.prepare_data_for_model import prepare_data

def test_knn_inference():
    df = pd.DataFrame([["sph","Firefox","Exécution d'un bouton(fr.infologic.core.client.modules.web.CopiloteWebSharedController)<DEFAUT>$AC$","t5","Ecriture memoire","t10"],
                       ["abc","Chrome","Affichage ecran","t7","Click souris","t20"],
                       ["tyu","Opera","Création d'un écran(infologic.crm.modules.CRM_ANNUAIRE.AnnuaireController)","t8","Exécution d'un bouton","t20"],
                       ["pou","Firefox","Création d'un écran(infologic.core.gui.controllers.nested.InputFormNestedWindow)","t5"],
                       ["abc","Chrome","Affichage ecran","t9","Affichage ecran","t22"],
                       ["pou","Firefox","Création d'un écran(infologic.core.gui.controllers.nested.InputFormNestedWindow)","t5"],
                       ]) 
    X_train,X_test,y_train,y_true = prepare_data(df=df)
    y_pred = knn_inference(X_train=X_train,Y_train=y_train,X_predict=X_test,number_of_neighbors=2)

    print("Xtest shape : ",X_test.shape)
    print("\n y test : ",y_true)
    print("\n y pred : ",y_pred)
    accuracy = f1_score(y_true,y_pred,average='macro')
    print("F1 score : ",accuracy)
    assert isinstance(accuracy,int)

def test_knn_f1_score():
    features_train = read_ds("data/train.csv")
    f1_score = knn_f1_score(features_train=features_train,nn=2)
    print("F1 score : ",f1_score)
    assert f1_score > 1




