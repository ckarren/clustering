import numpy as np
rng = np.random.default_rng(1234)
import pandas as pd
import os
import utils as ut

def run_clustering(data_file, n_clusters=2, feature_params={"annual":"all"}, save=False, plot=False, **kwargs):
    """ Perform clustering on water demand data and return a dataframe with the cluster id for each user in the dataset.
    Args:
        data_file: a .csv or .pkl file containing water demand data for a population of water users
        n_clusters: int or list of ints
        feature: dict or None describing what feature to cluster on. If None will default to "annual"
        save: bool whether or not to save the results to a .csv
        plot: bool whether or not to plot the means of each cluster of the results

    Returns:
        A dataframe of the cluster id for each user in the dataset.
        If save=True a .csv of the dataframe is saved. 
        If plot=True a plot is produced but is not saved. 
    
    """
    columns = ["User"]
    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]
    else:
        pass
    data_file_ext = os.path.splitext(data_file)[1]
    # use_df = pd.read_pickle(data_file)
    if data_file_ext == '.pkl':
        use_df = pd.read_pickle(data_file)
    elif data_file_ext == '.csv':
        use_df = pd.read_csv(data_file)    
    else:
        print("Unsupported data file. File must be a .csv or .pkl")
    use_df = use_df.sample(n=100, axis=1, random_state=1)
    use_df = ut.clean_outliers(use_df)
    
    results = []
    for key, value in feature_params.items():
        if key == "annual":
            X1_train = ut.groupby_year(use_df).T
            columns.append(str(key + "_all"))
            results.append(ut.perform_clustering(X1_train, n_clusters, **kwargs))
        elif key == "season":
            X1_train = ut.groupby_season(use_df).T
            if value == "all":
                columns.append(str(key + "_all"))
                results.append(ut.perform_clustering(X1_train, n_clusters, **kwargs))
            else:
                for season in value:
                    if season == 'summer':
                        X1_train = X1_train.iloc[0:24,:]
                    elif season == 'autumn':
                        X1_train = X1_train.iloc[24:48,:]
                    elif season == 'winter':
                        X1_train = X1_train.iloc[48:72,:]
                    elif season == 'spring':
                        X1_train = X1_train.iloc[72:97,:]
                    columns.append(str(key + "_" + season))
                    results.append(ut.perform_clustering(X1_train, n_clusters, **kwargs))
        # elif key == "month":
        #     X1_train = groupby_month(use_df).T
        #     if value == "all":
        #         results.append(perform_clustering(X1_train, n_clusters, **kwargs))
        #     else:
        #         for month in value:
        #             if month == "january":
        #                 X1_train = X1_train.iloc[0:24,:]
        #             if month == "february":
        #                 X1_train = X1_train.iloc[24:48,:]   
        #             if month == "march":
        #                 X1_train = X1_train.iloc[48:72,:]
        #             if month == "april":
        #                 X1_train = X1_train.iloc[72:96,:]
        #             if month == "may":
        #                 X1_train = X1_train.iloc[96:120,:]   
        #             if month == "june":
        #                 X1_train = X1_train.iloc[120:144,:]                       
        #             if month == "july":
        #                 X1_train = X1_train.iloc[144:168,:]
        #             if month == "august":
        #                 X1_train = X1_train.iloc[168:192,:]   
        #             if month == "september":
        #                 X1_train = X1_train.iloc[192:216,:]
        #             if month == "october":
        #                 X1_train = X1_train.iloc[216:240,:]
        #             if month == "november":
        #                 X1_train = X1_train.iloc[240:264,:]   
        #             if month == "december":
        #                 X1_train = X1_train.iloc[264:288,:]           
        #             results.append(perform_cluster(X1_train, n_clusters, **kwargs))
    final_results = [x for a in results for x in a]
    df = pd.DataFrame(list(zip(list(use_df.columns), *final_results)),
                      columns=columns)    
    print(df.head())
    if save:
        df.to_csv(f"{data_file[:-4]}_")
