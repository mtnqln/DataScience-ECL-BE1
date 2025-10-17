import pandas as pd
import numpy as np
import re
from .handle_data_pandas import read_ds
from typing import Dict
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

def browsers_per_player(df:pd.DataFrame, normalize:bool=False , unique:bool=True)->pd.DataFrame:
    """Output a DF with the number of browser per col, each row is an id"""
    """A row may look like this : 7, 0, 0, 0 if there is 4 possible browsers"""
    """If unique is set False then it returns id, 7, 0, 0, 0 ( because the id cannot be put as index if it's not unique )"""
    cols = get_browser_list(df)
    total_users: Dict[str,Dict[str,int]] = {} # Setting up a data structure with the name and counter-like dict for different browsers
    data = df.loc[:,:1]
    if unique==False:
        rows:list[Dict] = []
        rows_dict: Dict = {}
        for id_and_browser in data.loc[:,0:1].itertuples():
            rows_dict = {c: 0.0 for c in cols}
            rows_dict[0] = id_and_browser[1]
            rows_dict[id_and_browser[2]] = 1.0
            rows.append(rows_dict)
        df_out = pd.DataFrame(rows)
        df_out = df_out[[0, *cols]]
        return df_out
    
    for id_and_browser in data.loc[:,0:1].itertuples():
        if id_and_browser[1] not in total_users: 
            total_users[id_and_browser[1]] = {col:0 for col in cols}
        current_number = total_users[id_and_browser[1]].get(id_and_browser[2], 0)
        total_users[id_and_browser[1]][id_and_browser[2]] = current_number + 1
    df_browser = pd.DataFrame.from_dict(total_users,orient="index")
    if normalize==False:
        return df_browser
    # Normalize each row by its row-sum (avoid division by zero)
    row_sums = df_browser.sum(axis=1)
    row_sums = row_sums.replace(0, 1)
    return df_browser.div(row_sums, axis=0)

def get_normalize_browser_distribution(df:pd.DataFrame)->pd.Series:
    return get_Y_stats(df,1)

### Get mean time
def get_mean_time(df:pd.DataFrame)->pd.DataFrame:
    """Return id, mean time and total time for each session """
    df_time = df[[0]].copy()
    df_time["mean_time"] = None
    df_time["total_time"] = None
    for (index,row) in enumerate(df.itertuples()):
        last_element = str(row[-1])
        if isinstance(last_element, str) and re.match(r"^t\d{1,5}$", last_element.strip()):
                try:   
                    total_time = float(last_element.strip()[1:])
                except ValueError:
                    continue
        number_of_action:int = 0
        total=0
        for value in row[3:]:
            if isinstance(value,str) and not re.match(r"^t\d{1,5}$",str(value)):
                total += 1
        df_time.loc[index,"mean_time"] = float(total_time/total) if total > 0 else 0
        df_time.loc[index,"total_time"] = float(total_time)
        df_time["mean_time"] = df_time["mean_time"].astype(float)
        df_time["total_time"] = df_time["total_time"].astype(float)
        
    return df_time

### Get Action frequency
def clean_and_split_text(text: str) -> tuple[str, ...] | float:
    """Return list of cleaned splits or np.nan for None/'none'/empty."""
    if text is None:
        return np.nan
    s = str(text).strip()
    if s.lower() in {"none", "nan", ""}:
        return np.nan
    # Split on multiple delimiters and return all non-empty parts
    # split but keep the delimiter at the start of each following part
    splits = re.split(r'(?=(?:\(|<|\$|1))', s)
    parts = [p for p in splits if p.strip()]

    merged = []
    it = iter(parts)
    for p in it:
        if p.startswith('$') and not p.rstrip().endswith('$'):
            combined = p
            for q in it:
                combined += q
                if q.rstrip().endswith('$'):
                    break
            merged.append(combined)
        else:
            merged.append(p)

    cleaned = [s.strip() for s in merged if s.strip()]
    cleaned = tuple(cleaned)
    return cleaned if cleaned else np.nan

def parse_actions(df:pd.DataFrame)->tuple[list[list[str]],list[str]]:
    """Output tuple [ parsed actions , all possible actions ]"""
    """parsed actions looks like [id,[action1,action2...]]"""
    all_rows:list[list[str]] = []
    actions_list:list[str] = []
    for row in df.itertuples():
        id = row[1]
        buff:list[str] = []
        first_action = clean_and_split_text(row[3])
        if pd.notna(first_action) and isinstance(first_action,tuple): # type: ignore
            # all_rows.append([id]+[first_action[0]])
            buff.append(first_action[0])
            if first_action[0] not in actions_list:
                actions_list.append(str(first_action[0]))

        for value in row[4:]:
            if pd.notna(value):
                str_value = str(value)
                if not re.match(r"^t\d{1,5}$",str(str_value)) and re.match(r"^[A-Za-z]",str(str_value)):    
                        cleaned_text = clean_and_split_text(value)
                        if isinstance(cleaned_text,tuple):
                            buff.append(cleaned_text[0])
                            if cleaned_text[0] not in actions_list:
                                actions_list.append(cleaned_text[0])
        all_rows.append([id]+buff)

    return all_rows,actions_list

def get_actions_frequency(df:pd.DataFrame)->pd.DataFrame:
    """return a dataframe with each row as follow : id, number of action1, number of action2,..."""
    """the numbers are normalized on each rows"""
    all_rows, actions_list = parse_actions(df=df)
    actions_list.sort()
    new_row = []
    for (index,row) in enumerate(all_rows):
        user_id, *actions = row
        total = len(actions)
        counts = {}
        counts[0] = user_id #type: ignore
        for a in actions_list:
            counts[a] = (actions.count(a) / total) if total > 0 else 0
        serie = pd.Series(counts) 
        new_row.append(serie)
    df_actions = pd.DataFrame(new_row)
    return df_actions

# Frequency for consecutive actions


### Testing functions ###
if __name__=="__main__":
    ### Importing data
    features_train = read_ds("data/train.csv")
    features_test = read_ds("data/test.csv")
    print("TAIL : ",features_train.tail())
    ### Browser stat
    browser_list = get_browser_list(features_train)
    browsers = browsers_per_player(features_train)
    print("Browser counted :",browsers.head(),"\n")
    # We can see each people use only one browser
    print("Browser distribution : \n",get_normalize_browser_distribution(features_train).head(10),"\n")

    ### Y distribution
    print("Browser distrbution : \n",get_Y_stats(features_train,1),"\n")
    # print("Action : \n",get_Y_stats(features_train,3).head())

    ### To see if there is outliers
    outlier = get_outlier(features_train)
    # print("Outlier ranking : \n",outlier.head(10),"\n") # We can see there is no real outlier

    ### Mean time by action
    print("Mean time : \n",get_mean_time(features_train).head(10),"\n")

    ### Action frequency
    # df_frequency = get_actions_frequency(features_train)
    # print("Frequency : ",df_frequency.head(10))
    # print("Action frequency : \n",get_data_frequency(features_train).head(10),"\n")
