import pandas as pd
import re
from handle_data_pandas import read_ds

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

### Get Y stat ###
def get_Y_stats(df:pd.DataFrame,Y:int)->pd.Series:
    result = df[Y].value_counts(normalize=True)
    return result

### Detecting outliers
def get_outlier(df:pd.DataFrame)->pd.Series:
    df_for_outlier = df.notna().sum(axis=1) # Series with the position as index and the number of non Nan value
    return df_for_outlier.sort_values(ascending=False)

### Get mean time
def get_mean_time(df:pd.DataFrame)->pd.DataFrame:
    """Return mean time for each session"""
    df_mean_time = df[[0]].copy()
    df_mean_time["mean_time"] = None
    for (index,row) in enumerate(df.itertuples()):
        counter:int = 0
        mean:float = 0.0
        for value in row:
            if re.match(r"^t\d{1,2}$",str(value)):
                mean += sum(float(nbr) for nbr in value[1:])
                counter +=1
        df_mean_time.loc[index,"mean_time"] = float(mean/counter)
        
    return df_mean_time

### Get action frequency
def get_data_frequency():
    pass

### Testing functions ###
if __name__=="__main__":
    ### Importing data
    features_train = read_ds("data/train.csv")
    features_test = read_ds("data/test.csv")

    ### Browser stat
    print("Browser list : ",get_browser_list(features_train),"\n")
    browsers = browsers_per_player(features_train)
    print("Browser counted :",browsers.head(),"\n")
    # We can see each people use only one browser

    ### Y distribution
    print("Browser : \n",get_Y_stats(features_train,1).head(),"\n")
    # print("Action : \n",get_Y_stats(features_train,3).head())

    ### To see if there is outliers
    outlier = get_outlier(features_train)
    print("Outlier ranking : \n",outlier.head(10),"\n") # We can see there is no real outlier

    ### Mean time by action
    print("Mean time : \n",get_mean_time(features_train).head(10),"\n")


