#tests/unit/get_stat_from_data
import pandas as pd
from src.get_stat_from_data import browsers_per_player,get_mean_time,clean_and_split_text, parse_actions, get_actions_frequency

def test_browsers_per_player():
    df = pd.DataFrame([["sph","Firefox"],["abc","Chrome"],["tre","Firefox"],["sph","Firefox"]]) 
    result = browsers_per_player(df=df,normalize=True)
    expected_result = pd.DataFrame.from_dict({"sph":[1.0,0.0],"abc":[0.0,1.0],"tre":[1.0,0.0]}, orient='index', columns=["Firefox","Chrome"])
    pd.testing.assert_frame_equal(result,expected_result)

def test_get_mean_time():
    df = pd.DataFrame([["sph","Firefox","Exécution d'un bouton(fr.infologic.core.client.modules.web.CopiloteWebSharedController)<DEFAUT>$AC$","t5","Ecriture memoire","t10"],["abc","Chr\
    ome","Affichage ecran","t7","Click souris","t20"]])  
    result = get_mean_time(df=df)
    expected_result = pd.DataFrame([["sph",5.0,10.0],["abc",10.0,20.0]],columns=[0,"mean_time","total_time"])
    print(result.dtypes)
    print("expected result : ",expected_result.dtypes)
    pd.testing.assert_frame_equal(result,expected_result)

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
    assert result == ([["sph","Exécution d'un bouton","Exécution d'un bouton"],["abc","Affichage ecran"]],["Exécution d'un bouton","Affichage ecran"])

def test_get_actions_frequency():
    df = pd.DataFrame([["sph","Firefox","Exécution d'un bouton(fr.infologic.core.client.modules.web.CopiloteWebSharedController)<DEFAUT>$AC$","t5","Ecriture memoire","t10"],["abc","Chr\
    ome","Affichage ecran"]])  
    result = get_actions_frequency(df=df)

    expected_df = pd.DataFrame([["sph",0.0,0.5,0.5],["abc",1.0,0.0,0.0]], columns=[0,"Affichage ecran","Ecriture memoire","Exécution d'un bouton"])
    print(expected_df)
    print(result)
    pd.testing.assert_frame_equal(result, expected_df)

    


