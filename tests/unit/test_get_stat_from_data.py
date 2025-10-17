#tests/unit/get_stat_from_data
import pandas as pd
from src.get_stat_from_data import clean_and_split_text, parse_actions

def test_clean_and_split_text():
    text = "Exécution d'un bouton(fr.infologic.core.client.modules.web.CopiloteWebSharedController)<DEFAUT>$AC$"
    assert clean_and_split_text(text) == ("Exécution d'un bouton",
                                          "(fr.infologic.core.client.modules.web.CopiloteWebSharedController)",
                                          "<DEFAUT>",
                                          "$AC$"
                                          )
    
def test_parse_actions():
    df = pd.DataFrame([["sph","Firefox","Exécution d'un bouton(fr.infologic.core.client.modules.web.CopiloteWebSharedController)<DEFAUT>$AC$","t5","Exécution d'un bouton","t10"],["abc","Chr\
    ome","Affichage ecran"]])    
    result = parse_actions(df=df)   
    print(result)
    assert result == ([["sph",["Exécution d'un bouton","Exécution d'un bouton"]],["abc",["Affichage ecran"]]],["Exécution d'un bouton","Affichage ecran"])

