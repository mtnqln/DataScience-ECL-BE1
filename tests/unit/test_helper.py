import pandas as pd
from src.helper import normalize_df

def test_normalize_df():
    df = pd.DataFrame([[1,2,2],[3,4,4]])
    result = normalize_df(df,not_cols=0)
    expected_result = pd.DataFrame([[1,0.5,0.5],[3,0.5,0.5]])
    print("result : ",result)
    print("expected : ",expected_result)
    pd.testing.assert_frame_equal(result,expected_result)
