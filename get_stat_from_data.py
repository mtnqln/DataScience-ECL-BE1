import pandas as pd
import numpy as np
import re
from handle_data_pandas import read_ds

# Logs
import logging
logger = logging.getLogger(__name__)

### Get Y stat ###
def get_Y_stats(df:pd.DataFrame,Y:int)->pd.Series:
    result = df[Y].value_counts(normalize=True)
    return result

### Detecting outliers ###
def get_outlier(df:pd.DataFrame)->pd.Series:
    df_for_outlier = df.notna().sum(axis=1) # Series with the position as index and the number of non Nan value
    return df_for_outlier.sort_values(ascending=False)



### Data browser stat ###
def get_browser_list(df:pd.DataFrame)->list[str]:
    browser_list = []
    for browser in df[:][1]:
        if browser not in browser_list:
            browser_list.append(browser)
    return browser_list

def browsers_per_player(df:pd.DataFrame):
    cols = get_browser_list(df)
    total_users: dict[str,dict[str,int]] = {} # Setting up a data structure with the name and counter-like dict for different browsers
    data = df.loc[:,:1]
    for tuple in data.loc[:,0:1].itertuples():
        if tuple[1] not in total_users: 
            total_users[tuple[1]] = {col:0 for col in cols}
        else:
            current_number = total_users[tuple[1]].get(tuple[2], 0)
            total_users[tuple[1]][tuple[2]] = current_number + 1
    df_browser = pd.DataFrame.from_dict(total_users,orient="index")
    return df_browser

### Get mean time
def get_mean_time(df:pd.DataFrame)->pd.DataFrame:
    """Return mean time and total time for each session """
    df_time = df[[0]].copy()
    df_time["mean_time"] = None
    df_time["total_time"] = None
    for (index,row) in enumerate(df.itertuples()):
        counter:int = 0
        total:float = 0.0
        for value in row:
            if re.match(r"^t\d{1,5}$",str(value)):
                total += sum(float(nbr) for nbr in value[1:])
                counter +=1
        df_time.loc[index,"mean_time"] = float(total/counter)
        df_time.loc[index,"total_time"] = total
        
    return df_time

### Get somedata frequency
def clean_text(text:str)->float | str:
    """Return cleaned string or np.nan for None/'none'/empty."""
    if text is None:
        return np.nan
    s = str(text).strip()
    if s.lower() in {"none", "nan", ""}:
        return np.nan
    return s.split("(")[0].split("<")[0].split("$")[0].split("1")[0].strip()

def get_data_frequency(df:pd.DataFrame)->pd.DataFrame:
    actions_list:list = []
    all_rows = {}
    for row in df.itertuples():

        first_action = clean_text(row[3])
        if pd.notna(first_action):
            all_rows[row[0]] = [first_action]
            if first_action not in actions_list:
                actions_list.append(first_action)

        for value in row[4:]:
            str_value = str(value)
            if not re.match(r"^t\d{1,5}$",str(str_value)) and re.match(r"^[A-Za-z]",str(str_value)):    
                    cleaned_text = clean_text(value)
                    if pd.notna(cleaned_text):
                        all_rows[row[0]].append(cleaned_text)
                    if cleaned_text not in actions_list:
                        actions_list.append(cleaned_text)

    logger.info(f"Action list : {actions_list}")

    df_action = pd.DataFrame.from_dict(all_rows,orient="index")

    idxs = list(all_rows.keys())
    df_frequency = pd.DataFrame(0.0, index=idxs, columns=actions_list)

    # Fill proportions per row
    for rid, acts in all_rows.items():
        if not acts:
            continue
        vc = pd.Series(acts).value_counts(normalize=True)
        # assign only the columns present in vc
        df_frequency.loc[rid, vc.index] = vc.values

    return df_frequency

### Testing functions ###
if __name__=="__main__":
    logging.basicConfig(filename="get_stat_from_data.log",level=logging.INFO)
    ### Importing data
    features_train = read_ds("data/train.csv")
    features_test = read_ds("data/test.csv")

    ### Browser stat
    browser_list = get_browser_list(features_train)
    browsers = browsers_per_player(features_train)
    print("Browser counted :",browsers.head(),"\n")
    # We can see each people use only one browser

    ### Y distribution
    print("Browser distrbution : \n",get_Y_stats(features_train,1).head(),"\n")
    # print("Action : \n",get_Y_stats(features_train,3).head())

    ### To see if there is outliers
    outlier = get_outlier(features_train)
    print("Outlier ranking : \n",outlier.head(10),"\n") # We can see there is no real outlier

    ### Mean time by action
    # print("Mean time : \n",get_mean_time(features_train).head(10),"\n")

    ### Action frequency
    df_frequency = get_data_frequency(features_train)
    print("Frequency : ",df_frequency.head(10))
    # print("Action frequency : \n",get_data_frequency(features_train).head(10),"\n")



