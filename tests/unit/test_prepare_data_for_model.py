import pandas as pd
from src.prepare_data_for_model import prepare_data


def test_prepare_data():
    df = pd.DataFrame([["sph","Firefox","Exécution d'un bouton(fr.infologic.core.client.modules.web.CopiloteWebSharedController)<DEFAUT>$AC$","t5","Ecriture memoire","t10"],["abc","Chrome","Affichage ecran","t7","Click souris","t20"],["tyu","Opera","Création d'un écran(infologic.crm.modules.CRM_ANNUAIRE.AnnuaireController)","t8","Exécution d'un bouton","t20"]]) 
    
    (X_train,X_test,y_train,y_test) = prepare_data(df)

    assert X_train.shape == (2,10)
    assert y_train.shape == (2,3)