import pandas as pd
import glob
import numpy as np
from scipy import stats
from datetime import datetime, time, timedelta

# df2['average'] = df2.mean(numeric_only=True, axis=1) #create column of average use
# df2['total'] = df2.sum(axis=1)   #create column of cumulative use

def tot_col(dfx):
    dfx['Total'] = dfx.sum(axis=1) 
    return dfx

def pp_use(dfx, filter):
    dfx = dfx.reindex(filter, axis=1)
    dfx = dfx.loc[:, dfx.any()]
    dfx = tot_col(dfx)
    return dfx
    
def avg_col(dfx):
    dfx['AverageAll'] = dfx.mean(axis=1)
    return dfx

def unique(listx):
    uni_list = []
    for x in listx:
        if x not in uni_list:
            uni_list.append(x)
    return uni_list

def total_df(file):
    data = pd.read_csv(file, index_col=0, parse_dates=True)
    df = pd.DataFrame(data)
    xl = np.array([[x for x in df.sum().values]])
    df2 = pd.DataFrame(xl, columns = df.columns)
    return df2 

def conc_df(dfa, dfb):
    dfc = pd.concat([dfa, dfb])
    return dfc.fillna(0)

def bim_use(dfx):
    return dfx.sum(axis=0)

def billed_18(q):
    p = 17.69
    if q < 400:
        p += 0 
    elif q >= 400:
        p += (q - 400) / 100  * 3.50
    else:
        p = p 
    return round(p, 2) 

def billed_19(q):
    p = 18.40
    if q < 300:
        p += 0 
    elif 300 <= q < 400:
        p += (q - 300) / 100 * 2.89
    elif q >= 400: 
        p = p + 2.89 + (q - 400) / 100 * 3.50
    else:
        p = p 
    return round(p, 2)  

def qFilter(xcol, miudf, q):
    """returns: q number of lists of MiuIds for specified attribute 'xcol' """
    new_xcol = 'Q' + str(xcol)
    miudf = quant(miudf, xcol, q)
    quantile_list = [miu_desc_filter(miudf, new_xcol, i) for i in range(q)]
    return quantile_list

def dfFilter(qlist, usedf):
    """returns list of DataFrames"""
    dflist = [usedf.reindex(i, axis=1) for i in qlist]
    return dflist

def newDF(dflist):
    """returns 1 DataFrame with the average hourly demand of each quantile group"""
    dic = {}
    ind = dflist[0].index
    for i, item in enumerate(dflist):
        item['Average'] = item.mean(numeric_only=True, axis=1)
        k = 'q' + str(i+1) + 'ave'
        dic[k] = item['Average']

    new_df = pd.DataFrame(dic, 
                            index = ind)
    return new_df

def peakDF(dflist):
    """returns a list of lists of indexes of peak hourly values for each quantile  """
    pl = []
    for i, item in enumerate(dflist):
        tdf = tot_col(item)
        thu = tdf['Total']
        pl.append(thu.groupby(thu.index.date).idxmax())
    return pl

def miu_desc_filter(miu_dataframe, filter_type1, filter_value1):
    """
    miu_dataframe:: Pandas DataFrame
    filter_type:: Str
    filter_value:: Str
    """
    mdf = miu_dataframe.copy()
    filtered_data = mdf.loc[(mdf[filter_type1] == filter_value1)]#, mdf.index].to_list()
    data_list = [str(x) for x in filtered_data.index]
    return data_list

def miu_filter(miu_dataframe, filter_type, filter_value):
    """
    miu_dataframe: Pandas DataFrame
    filter_type: Str
    filter_value: Str
    ret_column: the column(s) to return in the filtered output
    """
    mdf = miu_dataframe.copy()
    filtered_data = mdf[(mdf[filter_type] == filter_value)]
    return filtered_data

def miu_range_filter(miu_dataframe, filter_type1, fil_val1_lo, fil_val1_hi):
    mdf = miu_dataframe.copy()
    filtered_data = mdf.loc[(mdf[filter_type1] > fil_val1_lo) & (mdf[filter_type1] < fil_val1_hi), 'MiuId'].to_list()
    data_list = [str(x) for x in filtered_data]
    return data_list

def combine_filters(fd1, fd2):
    c = [str(value) for value in fd1 if value in fd2]
    return c

#  def clean_outliers(df):
    #  print(df.shape)
    #  df = df[df.columns[(np.abs(stats.zscore(df, axis=1)) < 3).all(axis=0)]]
    #  a = df.max(axis=None)
    #  a = (np.abs(stats.zscore(df, axis=1)) < 3).all(axis=0)
    #  df = df.iloc[:,0:8]
    #  a = stats.zscore(df, axis=1)
    #  a.iloc[1,2] = 5
    #  print(a)
    #  print((np.abs(a) < 3).all(axis=0))
    #  b = (np.abs(a) < 3).all(axis=0)
    #  print(b)
    #  df = df[df.columns[b]]
    #  print(df.shape)
    #  return df
def clean_outliers(df):
    df = df[df.columns[~((df < 1.0).all(axis=0))]]
    df = df[df.columns[~((df > 400.0).any(axis=0))]]
    return df

def summary(df):
    a = stats.describe(df, axis=0)
    return a

#  def clean_outliers(dfx, lb=0.03, ub=400):
    #  dfx = dfx.loc[:, ((dfx != 0.0).any(axis=0))
                   #  & ((dfx > lb).any(axis=0))
                   #  & ((dfx < ub).all(axis=0))]
    #  return dfx

# for i in df_miu['MiuId']:
#     if i not in list(df_use.columns):
#         print(i)
#         df_miu.drop(df_miu.index[df_miu['MiuId'] == i], inplace=True)

# df_miu.to_csv('C:/Users/ckarren/OneDrive - North Carolina State University/Documents/DynamicPricing/dynamic_pricing/py_files+data/Python/CSV/lake_miu_lot_data_02.csv')


def q1(dfx):
    return dfx.quantile(0.25)

def q2(dfx):
    return dfx.quantile(0.5)

def q3(dfx):
    return dfx.quantile(0.75)

def quant(dfx, col, qn):
    new_col = 'Q' + str(col)
    dfx[new_col] = pd.qcut(dfx[col], qn, labels = False, duplicates = 'drop') 
    return dfx

def weekends(dfx):
    we = []
    for i in dfx.index:
        if i.dayofweek == 5:
            we.append(i.date())
    else:
        pass
    we = unique(we)
    we = [datetime.combine(x, time()) for x in we]
    wel = [x+timedelta(hours=47) for x in we]
    return we, wel

def add_time(n):
    n.index = pd.to_datetime(n.index)
    for i in n.index:
        s = n.xs(i)
        ni = i + timedelta(hours=23, minutes=59)
        n.loc[ni] = s
    n.sort_index(inplace=True)
    return n

def weekdays(dfx):
    return dfx[dfx.index.dayofweek < 5] 

def pickle_feature(input_path, output_path):
    for i in range(1,7):
        for j in range(1,3):
            file = f'hourly_use_SFR_y{j}_p{i}.pkl'
            df = pd.read_pickle(input_path + file)
            week_df = df.groupby([df.index.weekday, df.index.hour]).mean()
            print(f'Writing PDH_SFR_Y{j}P{i}.pkl file')
            week_df.to_pickle(output_path + f'PDH_SFR_Y{j}P{i}.pkl')

def split_week(df):
    weekdays = df.loc[df.index.weekday.isin([0,1,2,3,4])]
    weekends = df.loc[df.index.weekday.isin([5,6])]
    return weekdays, weekends

def groupby_month(df):
    weekdays, weekends = split_week(df)
  
    monthly_wd = weekdays.groupby([weekdays.index.month, 
                                   weekdays.index.hour]).mean()
    #  monthly_we = weekends.groupby([weekends.index.month,
                                   #  weekends.index.hour]).mean()
    #  monthly_we.index = monthly_we.index.map('Month: {0[0]} Hour: {0[1]}'.format)
 
    monthly_wd.reset_index(drop=True, inplace=True)
    #  monthly_we.reset_index(drop=True, inplace=True)

    return monthly_wd#, monthly_we

def groupby_year(df):
    weekdays, weekends = split_week(df)

    annual_wd = weekdays.groupby(weekdays.index.hour).mean()
    annual_we = weekends.groupby(weekends.index.hour).mean()

    annual_wd.reset_index(drop=True, inplace=True)
    annual_we.reset_index(drop=True, inplace=True)

    return annual_wd#, annual_we

def groupby_season(df):
    #  seasons = df.groupby(df.index).resample('QS-DEC').mean()
    summer = [6,7,8]
    autumn = [9,10,11]
    winter = [12,1,2]
    spring = [3,4,5]

    weekdays, weekends = split_week(df)

    summer_wd = weekdays.loc[weekdays.index.month.isin(summer)].copy()
    #  summer_we = weekends.loc[weekends.index.month.isin(summer)]
    spring_wd = weekdays.loc[weekdays.index.month.isin(spring)].copy()
    #  spring_we = weekends.loc[weekends.index.month.isin(spring)]
    autumn_wd = weekdays.loc[weekdays.index.month.isin(autumn)].copy()  
    #  autumn_we = weekends.loc[weekends.index.month.isin(autumn)]
    winter_wd = weekdays.loc[weekdays.index.month.isin(winter)].copy()  
    #  winter_we = weekends.loc[weekends.index.month.isin(winter)]

    summer_wd_avg = summer_wd.groupby(summer_wd.index.hour).mean()
    #  summer_we_avg = summer_we.groupby(summer_we.index.hour).mean()
    spring_wd_avg = spring_wd.groupby(spring_wd.index.hour).mean()
    #  spring_we_avg = spring_we.groupby(spring_we.index.hour).mean()
    autumn_wd_avg = autumn_wd.groupby(autumn_wd.index.hour).mean()
    #  autumn_we_avg = autumn_we.groupby(autumn_we.index.hour).mean()
    winter_wd_avg = winter_wd.groupby(winter_wd.index.hour).mean()
    #  winter_we_avg = winter_we.groupby(winter_we.index.hour).mean()

    by_season_wd = pd.concat([summer_wd_avg, 
                              autumn_wd_avg, 
                              winter_wd_avg,
                              spring_wd_avg])
    #  by_season_we = pd.concat([summer_we_avg,
                              #  autumn_we_avg,
                              #  winter_we_avg,
                              #  spring_we_avg])

    by_season_wd.reset_index(drop=True, inplace=True)
    #  by_season_we.reset_index(drop=True, inplace=True)

    return by_season_wd#, by_season_we
 
def analyse_dtw(n_clusters):
    """ Returns the number of members in each cluster, for radii 1-5"""
    li = []
    all_files = glob.glob(str('./4_DTW_results_scaled_r[1-5].csv'))
    for filename in all_files:
        df = pd.read_csv(
            filename,
            usecols=[0,1,2],
            header=0,
            index_col=0,
        )
        li.append(df)

    df = pd.concat(li, axis=1, ignore_index=True, verify_integrity=True) 
    df.drop(columns=[2,4,6,8], inplace=True) 
    df.rename(columns={
        0: 'User ID',
        1: 'r1',
        3: 'r2', 
        5: 'r3', 
        7: 'r4',
        9: 'r5'
    }, inplace=True)
    df.set_index('User ID', inplace=True)
    df['r2'] = df['r2'].apply(
        lambda x: 0 if x == 0 else (2 if x == 1 else (3 if x == 2 else 1)))
    df['r3'] = df['r3'].apply(
        lambda x: 2 if x == 3 else (3 if x == 2 else x))
    df['r4'] = df['r4'].apply(
        lambda x: 2 if x == 1 else (1 if x == 3 else (3 if x == 2 else 0)))
    df['r5'] = df['r5'].apply(
        lambda x: 0 if x == 2 else (2 if x == 1 else (1 if x == 0 else 3)))

    return df

#  n = df.apply(lambda x: x.value_counts())
#  def plot_clusters(df):
    #  fig = make_subplots(rows=1, cols=i)
    #  for yi in range(i):
        #  for xx in X_train[y_pred == yi]:
            #  fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]), y=xx.ravel(),
                                     #  line_color='grey'),
                          #  row=1, col=yi+1)
        #  fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]),
                                 #  y=km.cluster_centers_[yi].ravel(),
                                 #  line_color='darkred'),
                      #  row=1, col=yi+1)
    #  fig.show()
#  for yi in range(3):
#
#      plt.subplot(3,3,yi+4)
#      for xx in X_train[y_pred == yi]:
#          plt.plot(dba_km.cluster_centers_[yi].ravel(), 'r-')
#      if yi == 1:
#          plt.title('DBA $k$-means')
#
#  for yi in range(3):
#      plt.subplots(3,3,7+yi)
#      for xx in X_train[y_pred == yi]:
#          plt.plot(sdtw_km.cluster_centers_[yi].ravel(), 'r-')
#      if yi == 1:
#          plt.title('Soft-DTW $k$-means')
#  fig.update_layout(title_text='DTW k-means')
#  fig.show()

#  input_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')
#  water_use = pd.read_pickle(input_path + 'hourly_use_SFR_y1.pkl')
#  months = groupby_season(water_use)[0]
# print(months.columns)
