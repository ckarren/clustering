import glob
import numpy as np
rng = np.random.default_rng(1234)
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd 
import os


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
    return round(p * 1.03, 2) 

def billed_19(q):
    p = 18.40
    if q < 300:
        p += 0 
    elif 300 <= q < 400:
        p += (q - 300) / 100 * 2.89
    elif q >= 400: 
        p = p + 2.89 + (q - 400) / 100 * 3.50
    return round(p * 1.03, 2)  

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

def clean_outliers_sd(df):
    df = df[df.columns[(np.abs(stats.zscore(df, axis=1)) < 3).all(axis=0)]]
    a = df.max(axis=None)
    a = (np.abs(stats.zscore(df, axis=1)) < 3).all(axis=0)
    df = df.iloc[:,0:8]
    a = stats.zscore(df, axis=1)
    a.iloc[1,2] = 5
    b = (np.abs(a) < 3).all(axis=0)
    df = df[df.columns[b]]
    return df


def clean_outliers(df, lb=1.0, ub=400.0):
    df = df[df.columns[~((df < lb).all(axis=0))]]
    df = df[df.columns[~((df > ub).any(axis=0))]]
    return df

def summary(df):
    a = stats.describe(df, axis=0)
    return a

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

def groupby_year(df, op='mean'):
    weekdays, weekends = split_week(df)

    if op == 'mean':
        annual_wd = weekdays.groupby(weekdays.index.hour).mean()
        annual_we = weekends.groupby(weekends.index.hour).mean()
    elif op == 'total':
        annual_wd = weekdays.groupby(weekdays.index.hour).sum()
        annual_we = weekends.groupby(weekends.index.hour).sum()


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
    cluster_4 = {
        'r1': {
            0: 0,
            1: 1,
            2: 2,
            3: 3
            },
        'r2': {
            0: 0,
            1: 2,
            2: 3,
            3: 1
            },
        'r3': {
            0: 0,
            1: 1,
            2: 3,
            3: 2
        },
        'r4': {
            0:0,
            1:2,
            2:3,
            3:1

        },
        'r5': {
            0: 1,
            1: 2,
            2: 0,
            3: 3
        }
    }
    cluster_5 = {
        'r1': {
            0: 3,
            1: 2,
            2: 0,
            3: 1,
            4: 4
        },
        'r2': {
            0: 4,
            1: 2,
            2: 1,
            3: 0,
            4: 3
        },
        'r3': {
            0: 3,
            1: 2,
            2: 0,
            3: 1,
            4: 4
        },
        'r4': {
            0: 0,
            1: 2,
            2: 3,
            3: 1,
            4: 4
        },
        'r5': {
            0: 4,
            1: 1,
            2: 2,
            3: 0,
            4: 3
        }
    }
    rename_dicts = [cluster_4, cluster_5]
    li = []
    all_files = glob.glob(str(f'../RadiusComps/{n_clusters}_DTW_results_scaled_r[1-5].csv'))
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
    for i, column in enumerate(df):
        df[column] = df[column].map(rename_dicts[n_clusters-4][str(column)])
    return df

def plot_clusters(df):
    fig = make_subplots(rows=1, cols=i)
    for yi in range(i):
        for xx in X_train[y_pred == yi]:
            fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]), y=xx.ravel(),
                                     line_color='grey'),
                          row=1, col=yi+1)
        fig.add_trace(go.Scatter(x=np.arange(X_train.shape[1]),
                                 y=km.cluster_centers_[yi].ravel(),
                                 line_color='darkred'),
                      row=1, col=yi+1)
    fig.show()
    for yi in range(3):
        plt.subplot(3,3,yi+4)
        for xx in X_train[y_pred == yi]:
            plt.plot(dba_km.cluster_centers_[yi].ravel(), 'r-')
        if yi == 1:
            plt.title('DBA $k$-means')

    for yi in range(3):
        plt.subplots(3,3,7+yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(sdtw_km.cluster_centers_[yi].ravel(), 'r-')
        if yi == 1:
            plt.title('Soft-DTW $k$-means')
    fig.update_layout(title_text='DTW k-means')

def compare_radius_means(n_clusters, radii=['r1', 'r2', 'r3', 'r4', 'r5']):
    """ 
    produces plots to compare the means of clusters using the DTW algorithm with
    different radii of wrapping window
    n_clusters (int): the number of clusters used in the DTW algorithm
    radii (list): list of """

    df = analyse_dtw(n_clusters)
    df_use = pd.read_pickle('../InputFiles/y1_SFR_hourly.pkl') 
    df_use = clean_outliers(df_use)
    df_use = groupby_year(df_use)
    radii = radii
    clusters = list(range(n_clusters))
    n_radii = (len(radii))
    fontsize = 14

    fig, axs = plt.subplots(
            n_radii, n_clusters, 
            figsize=(24, 6), 
            sharex=True,
            sharey=True,
            layout='constrained')

    for ri, r in enumerate(radii):
        radius = df[r]
        for ci, c in enumerate(clusters):
            cluster = [str(x) for x in radius[radius == c].index.to_list()]
            df_use_rc =  df_use.filter(items=cluster)
            total = df_use_rc.sum(axis=1)
            average = df_use_rc.mean(axis=1)

            #  for i in df_use_rc.columns:
                #  i = str(i)
                #  ax.plot(df_use_rc.index, df_use_rc[i], c='grey')
            axs[ri, ci].plot(df_use_rc.index, average, c='crimson')
            if ri == 0:
                axs[ri, ci].set_title(f'Cluster {ci}', fontsize=fontsize)
            if ci == 0:
                axs[ri, ci].set_ylabel(f'Radius {ri+1}', fontsize=fontsize)
    fig.supxlabel('Time (hr)', fontsize=fontsize)
    fig.supylabel('Volume (gallons)', fontsize=fontsize)
    #  fig.suptitle(f'{n_clusters} Cluster Averages', fontsize=tfontsize)
    plt.show()


def calc_average_price():
    year1 = '../InputFiles/y1_SFR_hourly.pkl'
    use_y1 = pd.read_pickle(year1)
    df_y1 = pd.DataFrame(use_y1)
    df_y1 = clean_outliers(df_y1)
    year2 = '../InputFiles/y2_SFR_hourly.pkl'
    use_y2 = pd.read_pickle(year2)
    df_y2 = pd.DataFrame(use_y2)
    df_y2 = clean_outliers(df_y2)

    bill = pd.read_pickle('../InputFiles/bill_all.pkl')
    df_bill = pd.DataFrame(bill)
    df_bill.index = df_bill.index.map(str)

    #  summer_wd = weekdays.loc[weekdays.index.month.isin(summer)].copy()
    p1y1 = df_y1.loc[df_y1.index.month.isin([7, 8])].sum().round(3)
    p2y1 = df_y1.loc[df_y1.index.month.isin([9, 10])].sum().round(3)
    p3y1 = df_y1.loc[df_y1.index.month.isin([11, 12])].sum().round(3)
    p4y1 = df_y1.loc[df_y1.index.month.isin([1, 2])].sum().round(3)
    p5y1 = df_y1.loc[df_y1.index.month.isin([3, 4])].sum().round(3)
    p6y1 = df_y1.loc[df_y1.index.month.isin([5, 6])].sum().round(3)

    p1y2 = df_y2.loc[df_y2.index.month.isin([7, 8])].sum().round(3)
    p2y2 = df_y2.loc[df_y2.index.month.isin([9, 10])].sum().round(3)
    p3y2 = df_y2.loc[df_y2.index.month.isin([11, 12])].sum().round(3)
    p4y2 = df_y2.loc[df_y2.index.month.isin([1, 2])].sum().round(3)
    p5y2 = df_y2.loc[df_y2.index.month.isin([3, 4])].sum().round(3)
    p6y2 = df_y2.loc[df_y2.index.month.isin([5, 6])].sum().round(3)

    all_list = [p1y1, p2y1, p3y1, p4y1, p5y1, p6y1, p1y2, p2y2, p3y2, p4y2, p5y2, p6y2]

    for i, d in enumerate(all_list):
        all_list[i] = pd.concat([d, 
                                df_bill.iloc[:,i]],
                                 axis=1,
                                join='inner')
        all_list[i].columns = ['Q', 'P']
        all_list[i]['P_ave'] = all_list[i]['P'] / all_list[i]['Q']
    
    q_all = pd.concat(all_list, axis=0, join='inner')
    q_all = q_all.reset_index(names='user')
    q_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    q_all.dropna(axis=0, how='any', inplace=True, ignore_index=True)
    q_all.to_pickle('average_price_data.pkl')

def instruments():
   period_index = pd.period_range(start='2018-05-01', end='2020-06-01', freq='2M')
   datetime_index = period_index.to_timestamp()
   FC = ([17.69] * 6) + ([18.40] * 6)
   block1 = ([0.0] * 6) + ([2.89] * 6)
   blockdiff1 = ([0.0] * 6) + ([2.89] * 6)
   blockdiff2 = ([3.5] * 6) + ([0.61] * 6)
   DOS = datetime_index.to_series().diff()[1:].astype('int64') / (864 * 10**11)

   data = {
       'FC': FC,
       'block1': block1,
       'blockdiff1': blockdiff1,
       'blockdiff2': blockdiff2,
       'DOS': DOS
   }
   df = pd.DataFrame(data, index=datetime_index[1:])
   return df

def weather_data():
    file1 = '3082341.csv'
    file2 = 'monthly.csv'

    data1 = pd.read_csv(
        file1, 
        usecols=[
            'DATE',
            'REPORT_TYPE',
            'DailyAverageRelativeHumidity',
            'DailyPrecipitation',
            'MonthlyDaysWithGT001Precip',
            'MonthlyDaysWithGT90Temp',
            'MonthlyMeanTemperature',
            'MonthlyMinimumTemperature',
            'MonthlyTotalLiquidPrecipitation',
            'NormalsCoolingDegreeDay',
            'MonthlyGreatestPrecip'
        ],
        index_col='DATE',
        parse_dates=True,
        na_values=['T'],
        #  na_values={'DailyPrecipitation': 'T'},
        #  low_memory=False
        dtype={
            'REPORT_TYPE': str,
            'DailyAverageRelativeHumidity': np.float64,
            'MonthlyDaysWithGT001Precip': np.float64,
            'MonthlyDaysWithGT90Temp': np.float64,
            'MonthlyMeanTemperature': np.float64,
            'MonthlyMinimumTemperature': np.float64,
            'MonthlyTotalLiquidPrecipitation': np.float64,
            'NormalsCoolingDegreeDay': np.float64,
            'MonthlyGreatestPrecip': np.float64
        }
    )

    data2 = pd.read_csv(
        file2,
        usecols=[
        'Month Year',
        'Total ETo (in)'],
        index_col='Month Year',
        parse_dates=True,
        date_format='%b %Y'
    )
    df = pd.DataFrame(data1)
    df.index = df.index.to_period('M').to_timestamp() 
    #  df.index.replace(day=1)
    ET = pd.DataFrame(data2)
    ET = ET.resample('MS').mean()
    rhave = df.resample('MS')['DailyAverageRelativeHumidity'].mean().round(2)
    pp = df.resample('MS')['DailyPrecipitation'].mean().round(2)
    df = df[df['REPORT_TYPE'] == 'SOM  ']
    df = df.drop(labels=[
        'REPORT_TYPE',
        'DailyAverageRelativeHumidity',
        'DailyPrecipitation'],
     axis=1)
    df['rhave'] = rhave
    df['pp'] = pp
    df['ET'] = ET
    df.dropna(axis=0, how='all', inplace=True)
    df.rename(columns={
            'MonthlyDaysWithGT90Temp': 'gt90',
            'MonthlyDaysWithGT001Precip': 'ppdays',
            'MonthlyGreatestPrecip': 'pmax',
            'MonthlyMeanTemperature': 'tave',
            'MonthlyMinimumTemperature': 'tmin',
            'MonthlyTotalLiquidPrecipitation': 'totalpp',
            'NormalsCoolingDegreeDay': 'cdd'
            },
        inplace=True)
    df = df.resample('2MS', origin='start').agg({
            'ppdays': 'sum',
            'gt90': 'sum',
            'pmax': 'max',
            'tave': 'mean',
            'tmin': 'min',
            'totalpp': 'sum',
            'cdd': 'sum',
            'rhave': 'mean',
            'pp': 'mean',
            'ET': 'mean'
            })
    df.to_pickle('weather_data.pkl')


def prepare_regression(sample=False, **kwargs):

    year1 = '../InputFiles/y1_SFR_hourly.pkl'
    use_y1 = pd.read_pickle(year1)
    df_y1 = pd.DataFrame(use_y1)
    df_y1 = clean_outliers(df_y1)
    year2 = '../InputFiles/y2_SFR_hourly.pkl'
    use_y2 = pd.read_pickle(year2)
    df_y2 = pd.DataFrame(use_y2)
    df_y2 = clean_outliers(df_y2)

    bill = pd.read_pickle('../InputFiles/bill_all.pkl')
    df_bill = pd.DataFrame(bill)
    df_bill.index = df_bill.index.map(str)

    weather = pd.read_pickle('weather_data.pkl')
    df_weather = pd.DataFrame(weather)

    instruments = instruments()
    #  summer_wd = weekdays.loc[weekdays.index.month.isin(summer)].copy()
    p1y1 = df_y1.loc[df_y1.index.month.isin([7, 8])].sum().round(3)
    p2y1 = df_y1.loc[df_y1.index.month.isin([9, 10])].sum().round(3)
    p3y1 = df_y1.loc[df_y1.index.month.isin([11, 12])].sum().round(3)
    p4y1 = df_y1.loc[df_y1.index.month.isin([1, 2])].sum().round(3)
    p5y1 = df_y1.loc[df_y1.index.month.isin([3, 4])].sum().round(3)
    p6y1 = df_y1.loc[df_y1.index.month.isin([5, 6])].sum().round(3)

    p1y2 = df_y2.loc[df_y2.index.month.isin([7, 8])].sum().round(3)
    p2y2 = df_y2.loc[df_y2.index.month.isin([9, 10])].sum().round(3)
    p3y2 = df_y2.loc[df_y2.index.month.isin([11, 12])].sum().round(3)
    p4y2 = df_y2.loc[df_y2.index.month.isin([1, 2])].sum().round(3)
    p5y2 = df_y2.loc[df_y2.index.month.isin([3, 4])].sum().round(3)
    p6y2 = df_y2.loc[df_y2.index.month.isin([5, 6])].sum().round(3)
#
    all_list = [p1y1, p2y1, p3y1, p4y1, p5y1, p6y1, p1y2, p2y2, p3y2, p4y2, p5y2, p6y2]

    if kwargs:
        if kwargs['price'] == 'total':
            for i, d in enumerate(all_list):
                all_list[i] = pd.concat([d.apply(np.log).round(3),
                                        df_bill.iloc[:,i].apply(np.log).round(3)],
                                        axis=1,
                                        join='inner')
                all_list[i]['period'] = str((i % 6) + 1)
                all_list[i].columns = ['logQ', 'logP', 'period']
        elif kwargs['price'] == 'lagged average':
            for i, d in enumerate(all_list):
                if i > 0:
                    all_list[i] = pd.concat([d,
                                            df_bill.iloc[:,i-1]],
                                             axis=1,
                                            join='inner')
                    all_list[i].columns = ['Q', 'P']
                    all_list[i]['period'] = str((i % 6) + 1)
                    all_list[i]['P_ave'] = np.log(all_list[i]['P'] /
                                                  all_list[i]['Q']).round(3)
                    all_list[i]['Q'] = all_list[i]['Q'].apply(np.log).round(3)
                    all_list[i] = all_list[i].assign(**df_weather.iloc[i,:])
                    all_list[i] = all_list[i].assign(**instruments.iloc[i-1,:])

    q_all = pd.concat(all_list[1:-1], axis=0, join='inner')
    q_all = q_all.reset_index(names='user')
    q_all.replace([np.inf, -np.inf, 'T'], np.nan, inplace=True)
    q_all.dropna(axis=0, how='any', inplace=True, ignore_index=True)
    if sample:
        if not kwargs:
            n_sample = 300
        else:
            n_sample = kwargs['n_sample']
        x = rng.choice(pd.unique(q_all['user']), n_sample, replace=False)
        q_all = q_all.loc[q_all['user'].isin(x)]
        q_all.to_pickle(f'reg_data_{n_sample}.pkl')
    else:
        q_all.to_pickle('LAP_inst_reg_data.pkl')

prepare_regression()

def add_dummies(file='reg_data.pkl'):
    file = file
    data = pd.read_pickle(
        file
    )
    clusters_file = '../RadiusComps/4_DTW_results_scaled_r2.csv'
    clusters = pd.read_csv(
        clusters_file, 
        usecols=)
    data = pd.get_dummies(
        data=data,
        columns=["user", "period"],
        drop_first=True,
        dtype=int
    )
    data.to_pickle(f'{file[:-4]}_with_dummies.pkl')


def users():
    users = pd.read_pickle('../InputFiles/user_ids.pkl')
    users = [int(x) for x in users]
    print(np.min(users))


def load_reg_data(sample=False, **kwargs):
    if sample:
        if not kwargs:
            n_sample = 300
        else:
            n_sample = kwargs['n_sample']
        data_file = f'reg_data_{n_sample}_with_dummies.pkl'
        if os.path.exists(data_file):
            data_file = data_file
        else:
            prepare_regression(sample=True, n_sample=n_sample)
            add_dummies('reg_data_' + str(n_sample) + '.pkl')
            data_file = data_file
    else:
        data_file = 'lagged_average_price_reg_data_with_dummies.pkl'

    return data_file


