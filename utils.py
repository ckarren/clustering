import glob
import matplotlib.lines as mlines
from sklearn.preprocessing import normalize
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
rng = np.random.default_rng(1234)
import statsmodels.api as sm
#  from statsmodels.sandbox.regression.gmm import IV2SLS
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import HBoxDivider, VBoxDivider
import mpl_toolkits.axes_grid1.axes_size as Size
import matplotlib.ticker as ticker
import pandas as pd 
import os

cluster_colors = ['cornflowerblue',
                  'darkorange',
                  'forestgreen',
                  'tomato',
                  'mediumorchid']

cluster_colors_dict = {'DLN': 'cornflowerblue',
                       'SMD': 'darkorange',
                       'DEM': 'forestgreen',
                       'DE': 'tomato',
                       'DM': 'mediumorchid'}

# for dashboard 
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

    new_df = pd.DataFrame(dic, index = ind)
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

def clean_outliers(df, lb=1.0, ub=400.0, ll=-10.0):
    df = df[df.columns[~((df < lb).all(axis=0))]]
    df = df[df.columns[~((df > ub).any(axis=0))]]
    df = df[df.columns[~((df < ll).any(axis=0))]]
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

def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform-ax.transAxes,
            ha='center', va='center', fontsize=fontsize, color='black')

# for clustering:
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
    #  autumn_we = weekends.loc[weekends.index.month.isin(spring)]
    winter_wd = weekdays.loc[weekdays.index.month.isin(winter)].copy()  
    #  winter_we = weekends.loc[weekends.index.month.isin(spring)]

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

def analyse_dtw(n_clusters, n_radius):
    """ Returns the number of members in each cluster, for radii n_radius
    n_clusters (int): 
    n_radius (list): the radii used
    """
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
    cluster_6 = {
        'r1': {
            0: 3,
            1: 2,
            2: 0,
            3: 1,
            4: 4,
            5: 5
        },
        'r2': {
            0: 4,
            1: 2,
            2: 1,
            3: 0,
            4: 3,
            5: 5
        },
        'r3': {
            0: 3,
            1: 2,
            2: 0,
            3: 1,
            4: 4,
            5: 5
        }
    }
    cluster_7 = {
        'r1': {
            0: 3,
            1: 2,
            2: 0,
            3: 1,
            4: 4,
            5: 5,
            6: 6
        },
        'r2': {
            0: 4,
            1: 2,
            2: 1,
            3: 0,
            4: 3,
            5: 5,
            6: 6

        },
        'r3': {
            0: 3,
            1: 2,
            2: 0,
            3: 1,
            4: 4,
            5: 5,
            6: 6
        }
    }
    rename_dicts = [cluster_4, cluster_5, cluster_6, cluster_7]
    li = []
    all_files = glob.glob(str(f'../RadiusComps/{n_clusters}_DTW_results_scaled_r[1-5].csv'))
    for filename in all_files:
        df = pd.read_csv(
            filename,
            usecols=[1,2],
            header=0,
            index_col=0,
        )
        li.append(df)

    df = pd.concat(li, axis=1, ignore_index=True, verify_integrity=True)
    df.rename(columns={
        0: 'r1',
        1: 'r2',
        2: 'r3',
        3: 'r4',
        4: 'r5'
    }, inplace=True)
    df.set_index('User', inplace=True)
    for i, column in enumerate(df):
        df[column] = df[column].map(rename_dicts[n_clusters-4][str(column)])
    return df

def compare_radius_means(n_clusters, radii=['r1', 'r2', 'r3', 'r4', 'r5']):
    """ 
    produces plots to compare the means of clusters using the DTW algorithm with
    different radii of wrapping window
    n_clusters (int): the number of clusters used in the DTW algorithm
    radii (list): list of radii to compare
    """

    df = analyse_dtw(n_clusters, radii)
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

def cluster_hourly_heatmap():
    file = '../5_clusters_output/total_hourly_use_by_cluster.csv'
    df = pd.read_csv(file, header=0, index_col=0)
    df['Total'] = df.sum(axis=1)
    df_normal = df / df.max()
    df_normal = df_normal.T
    df = df.T
    clusters = df_normal.iloc[:-1,:]
    total = df_normal.iloc[-1:,:]
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    #  ax2.pcolormesh(clusters)
    im1 = axs[0].imshow(clusters) 
    #  ax2.xaxis.set_ticklabels([])
    #  ax2.yaxis.set_ticklabels(['1', '2', '3', '4', '5'])
    #  ax1.set_yticks(np.arange(len(df.index)-1), ['1', '2', '3', '4', '5'])
    axs[0].set_yticks(np.arange(len(df.index)-1), ['DLN', 'SMD', 'DEM', 'DE',
                                                'DM'])
    axs[0].set_ylabel('Cluster', fontsize=16)
    axs[0].tick_params(labelsize=14)
    axs[0].xaxis.set_ticklabels([])
    #  ax1.pcolormesh(total)
    im2 = axs[1].imshow(total)
    axs[1].yaxis.set_ticklabels([])
    axs[1].tick_params(labelsize=14)
    axs[1].set_xlabel('Time (hr)', fontsize=16)
    axs[1].set_ylabel('Total', fontsize=16)

    #  divider = VBoxDivider(
        #  fig, 111,
        #  horizontal=[Size.AxesX(ax1), Size.Scaled(1), Size.AxesX(ax2)],
        #  vertical=[Size.AxesY(ax1), Size.Fixed(pad), Size.AxesY(ax2)])
#
    #  ax1.set_axes_locator(divider.new_locator(0))
    #  ax2.set_axes_locator(divider.new_locator(2))
    #  fig.colorbar(im1, ax=ax1, location='bottom', label='Volume (gallons)')
    cbar = fig.colorbar(im1, ax=axs[1], orientation='horizontal')
    cbar.ax.tick_params(labelsize=14)
    plt.show()

def cluster_hourly_heatmap_ind(n_clusters, radius):
    cluster_file = f'../RadiusComps/{n_clusters}_DTW_results_scaled_r{radius}.csv'
    #  cluster_file = f'../RadiusComps/{n_clusters}_euclidean_results_dtw{radius}.csv'
    df_cluster = pd.read_csv(cluster_file,
                             usecols=[1,2],
                            header=0,
                            index_col=0
                            )
    cluster_cust = []
    clusters = list(range(n_clusters))
    for cluster in clusters:
        df_cluster1 = df_cluster[df_cluster['DBA cluster'] == cluster]
        cluster_cust.append(str(np.random.choice(df_cluster1.index.values)))

    df_use1 = pd.read_pickle('../InputFiles/y1_SFR_hourly.pkl') 
    df_use2 = pd.read_pickle('../InputFiles/y2_SFR_hourly.pkl')
    df_use = pd.concat([df_use1, df_use2], join='inner')
    df_use = clean_outliers(df_use)

    df_use = df_use.filter(cluster_cust)
    clusters = list(range(n_clusters))
    cluster_array = []
    for col in df_use:
        a = df_use[col].values.reshape((730,24))
        a = normalize(a, norm='max')
        cluster_array.append(a)
    cluster_names = ['Dominant Late Night',
                     'Pronounced Late Morning',
                     'Dominant Early Morning',
                     'Dominant Evening',
                     'Dominant Morning']

    fig, axs = plt.subplots(nrows=1, ncols=n_clusters)
    aw = .2
    ah = .8
    margin = 0.05
    #  if period == 'year':
        #  df_use = groupby_year(df_use)
    #  elif period == 'month':
        #  df_use = groupby_month(df_use)
    #  elif period == 'season':
        #  df_use = groupby_season(df_use)


    for c in clusters:
        #  cluster = [str(x) for x in df_cluster[df_cluster['k-means cluster'] == c].index.to_list()]
        #  cluster = [str(x) for x in df_cluster[df_cluster['DBA cluster'] == c].index.to_list()]
        #  df_use_c =  normalized_use.filter(items=cluster)
        #  df_use_c = df_use_c.T
        im = axs[c].imshow(cluster_array[c], aspect='auto')
        axs[c].tick_params(axis='x', labelsize=10)
        axs[c].yaxis.set_ticklabels([])
        axs[c].xaxis.set_major_locator(ticker.FixedLocator([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]))
        axs[c].xaxis.set_ticklabels(['00:00','01:00', '02:00', '03:00', '04:00',
                                     '05:00', '06:00', '07:00', '08:00',
                                     '09:00', '10:00', '11:00', '12:00',
                                     '01:00', '02:00', '03:00', '04:00',
                                     '05:00', '06:00', '07:00', '08:00',
                                     '09:00', '10:00', '11:00'],
                                    rotation='vertical')
        axs[c].annotate(f'{cluster_names[c]}',
                        xy=(0.50, 1.03), xycoords='axes fraction',
                        ha='center', va='top',
                        fontsize=14
                        )
        if c == 4:
            cbar = axs[c].figure.colorbar(im, ax=axs[c], location='right',
                                          orientation='vertical')
    plt.show()

def cluster_hourly_heatmap_all(n_clusters, radius, period='year'):
    cluster_file = f'../RadiusComps/{n_clusters}_DTW_results_scaled_r{radius}.csv'
    #  cluster_file = f'../RadiusComps/{n_clusters}_euclidean_results_dtw{radius}.csv'
    df_cluster = pd.read_csv(cluster_file,
                             usecols=[1,2],
                            header=0,
                            index_col=0
                            )
    df_use1 = pd.read_pickle('../InputFiles/y1_SFR_hourly.pkl') 
    df_use2 = pd.read_pickle('../InputFiles/y2_SFR_hourly.pkl')
    df_use = pd.concat([df_use1, df_use2], join='inner')
    df_use = clean_outliers(df_use)

    clusters = list(range(n_clusters))
    cluster_names = ['Dominant Late Night',
                     'Steady Mid-Day',
                     'Dominant Early Morning',
                     'Dominant Evening',
                     'Dominant Morning']

    fig, axs = plt.subplots(nrows=1, ncols=n_clusters)
    aw = .2
    ah = .8
    margin = 0.05

    if period == 'year':
        df_use = groupby_year(df_use)
    elif period == 'month':
        df_use = groupby_month(df_use)
    elif period == 'season':
        df_use = groupby_season(df_use)

    normalized_use = df_use / df_use.max() 

    for c in clusters:
        #  cluster = [str(x) for x in df_cluster[df_cluster['k-means cluster'] == c].index.to_list()]
        cluster = [str(x) for x in df_cluster[df_cluster['DBA cluster'] == c].index.to_list()]
        df_use_c =  normalized_use.filter(items=cluster)
        df_use_c = df_use_c.T
        im = axs[c].imshow(df_use_c, aspect='auto')
        axs[c].tick_params(axis='x', labelsize=12)
        axs[c].yaxis.set_ticklabels([])
        #  axs[c].xaxis.set_major_locator(ticker.FixedLocator([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]))
        axs[c].xaxis.set_major_locator(ticker.FixedLocator([0,4,8,12,16,20,23]))
        #  axs[c].xaxis.set_ticklabels(['00:00','01:00', '02:00', '03:00', '04:00',
                                     #  '05:00', '06:00', '07:00', '08:00',
                                     #  '09:00', '10:00', '11:00', '12:00',
                                     #  '01:00', '02:00', '03:00', '04:00',
                                     #  '05:00', '06:00', '07:00', '08:00',
                                     #  '09:00', '10:00', '11:00'],
                                    #  rotation='vertical')
        axs[c].xaxis.set_ticklabels(['00:00', '04:00','08:00', '12:00', '04:00', '08:00', '11:00'], rotation='vertical')
        axs[c].annotate(f'{cluster_names[c]}',
                        xy=(0.50, 1.03), xycoords='axes fraction',
                        ha='center', va='top',
                        fontsize=16
                        )
        if c == 4:
            cbar = axs[c].figure.colorbar(im, ax=axs[c], location='right',
                                          orientation='vertical')
            cbar.ax.tick_params(labelsize=16)
    plt.show()

def cluster_summary(n_clusters, radius, **kwargs): 
    """ 
    produces plots to compare the means of clusters using the DTW algorithm with
    different radii of wrapping window
    n_clusters (int): the number of clusters used in the DTW algorithm
    radii (list): list of radii to compare
    """

    cluster_file = f'../RadiusComps/{n_clusters}_DTW_results_scaled_r{radius}.csv'
    df_cluster = pd.read_csv(cluster_file,
                             usecols=[1,2],
                            header=0,
                            index_col=0
                            )
    df_use1 = pd.read_pickle('../InputFiles/y1_SFR_hourly.pkl') 
    df_use2 = pd.read_pickle('../InputFiles/y2_SFR_hourly.pkl')
    df_use_all = pd.concat([df_use1, df_use2], join='inner')
    df_use1 = clean_outliers(df_use1)
    df_use2 = clean_outliers(df_use2)
    df_use_all = clean_outliers(df_use_all)

    #  if kwargs:
        #  if kwargs['period'] == 'year':
            #  df_use_all = groupby_year(df_use_all)
            #  df_use_1 = groupby_year(df_use_1)
            #  df_use_2 = groupby_year(df_use_2)
        #  elif kwargs['period'] == 'month':
            #  df_use_all = groupby_month(df_use_all)
            #  df_use_1 = groupby_month(df_use_1)
            #  df_use_2 = groupby_month(df_use_2)
        #  elif kwargs['period'] == 'season':
            #  df_use_all = groupby_season(df_use_all)
            #  df_use_1 = groupby_season(df_use_1)
            #  df_use_2 = groupby_season(df_use_2)
        #  else:
            #  print('keyword period must be one of "year", "month", or "season".')
    clusters = list(range(n_clusters))
    fontsize = 18
    #  colors =
    cluster_names = ['Dominant Late Night',
                     'Steady Mid-Day',
                     'Dominant Early Morning',
                     'Dominant Evening',
                     'Dominant Morning']
    total = {}
    average = {}
    for name in cluster_names:
        total[name] = {}
        average[name] = {}

    for c in clusters:
        cluster = [str(x) for x in df_cluster[df_cluster['DBA cluster'] == c].index.to_list()]
        df_use_all_c =  df_use_all.filter(items=cluster)
        df_use_y1_c =  df_use1.filter(items=cluster)
        df_use_y2_c =  df_use2.filter(items=cluster)
        #  total[f'{cluster_names[c]}']['All'] = np.round(
                                                #  df_use_all_c.sum(axis=1).sum(axis=0), 3)
        #  total[f'{cluster_names[c]}']['Y1'] = np.round(
                                                #  df_use_y1_c.sum(axis=1).sum(axis=0), 3)
        #  total[f'{cluster_names[c]}']['Y2'] = np.round(
                                                #  df_use_y2_c.sum(axis=1).sum(axis=0), 3)
        average[f'{cluster_names[c]}']['All'] = np.round(
                                                df_use_all_c.mean(axis=1).mean(), 3)
        average[f'{cluster_names[c]}']['Y1'] = np.round(
                                                df_use_y1_c.mean(axis=1).mean(), 3)
        average[f'{cluster_names[c]}']['Y2'] = np.round(
                                                df_use_y2_c.mean(axis=1).mean(), 3)
        #  average.to_csv(f'{n_clusters}_c{c}_average_all2.csv')
        #  total.to_csv(f'{n_clusters}_c{c}_total_all2.csv')
    #  total_df = pd.DataFrame(total)
    average_df = pd.DataFrame(average)
    #  total_df.to_csv('total_use_by_cluster.csv')
    average_df.to_csv('average_use_by_cluster.csv')

def plot_inertia():
    k = [2, 3, 4, 5, 6, 7, 8, 9]
    inertia = [16.47, 14.38, 13.05, 12.00, 11.02, 10.38, 9.86, 9.40]
    inertia2 = [8.66, 6.89, 6.09, 5.31, 5.03, 4.72, 4.49, 4.34]
    fig, ax = plt.subplots()
    ax.plot(k, inertia2, 'o-', linewidth=3)
    ax.set_xlabel('k', fontsize=18)
    ax.set_ylabel('Inertia', fontsize=18)
    ax.tick_params(labelsize=14)
    plt.show()
def join_seasons():
    summer_data = pd.read_csv('../5_clusters_output/5_clusters_DTW_results_scaled_summer.csv',
                            usecols=[1,2],
                            index_col=0, header=0)
    summer_df = pd.DataFrame(summer_data)

    winter_data = pd.read_csv('../5_clusters_output/5_clusters_DTW_results_scaled_winter.csv',
                            usecols=[1,2],
                            index_col=0, header=0)
    winter_df = pd.DataFrame(winter_data)
    annual_data = pd.read_csv('../RadiusComps/5_DTW_results_scaled_r1.csv',
                              usecols=[1,2],
                              index_col=0, 
                              header=0)
    annual_df = pd.DataFrame(annual_data)
    df = pd.concat([summer_df, winter_df], axis=1, join='inner')
    rename_clusters = {
        'winter cluster' : {
            0: 1,
            1: 2,
            2: 0,
            3: 3,
            4: 4
        },
        'summer cluster': {
            0: 0,
            1: 2,
            2: 1,
            3: 3,
            4: 4
        }
    }
    for column in df:
        df[column] = df[column].map(rename_clusters[column])
    tot_df = pd.concat([df, annual_df], axis=1, join='inner')

    tot_df.to_csv('5_clusters_summer_winter_all.csv')

def rename_seasonal_cluster():
    data = pd.read_csv('5_clusters_by_season.csv', index_col=0)
    df = pd.DataFrame(data)
    rename_clusters = {
        'summer Cluster': {
            0: 1,
            1: 4,
            2: 2,
            3: 0,
            4: 3
        },
        'autumn Cluster': {
            0: 4,
            1: 3, 
            2: 1,
            3: 2,
            4: 0
        },
        'winter Cluster': {
            0: 0,
            1: 2,
            2: 3,
            3: 1,
            4: 4
        },
        'spring Cluster': {
            0: 0,
            1: 1,
            2: 3,
            3: 4,
            4: 2
        }
    }

    for column in df:
        df[column] = df[column].map(rename_clusters[column])

    df.to_csv('rename_clusters_by_season.csv')
    return df

def by_period(period, df_use, axs):
    if period == 'year':
        df_use = groupby_year(df_use)
        for ax in axs:
            ax.set_xlim([0,23])
        #  ax.set_ylim([0,45.0])
    elif period == 'month':
        df_use = groupby_month(df_use)
        for ax in axs:
            ax.set_xlim([0,287])
        #  ax.set_ylim([0,45.0])
    elif period == 'season':
        df_use = groupby_season(df_use)
        for ax in axs:
            ax.set_xlim([0,95])
    return df_use

def by_season(season, df_use, axs):
    if season == 'summer':
        for ax in axs:
            #  ax.set_ylim([0,45.0])
            ax.set_xlim([0,23])
        df_use = df_use.iloc[0:24,:]
        return df_use
    elif season == 'autumn':
        for ax in axs:
            #  ax.set_ylim([0,35.0])
            ax.set_xlim([0,23])
        df_use = df_use.iloc[24:48,:]
        df_use.reset_index(drop=True, inplace=True)
        return df_use
    elif season == 'winter':
        for ax in axs:
            #  ax.set_ylim([0,35.0])
            ax.set_xlim([0,23])
        df_use = df_use.iloc[48:72,:]
        df_use.reset_index(drop=True, inplace=True)
        return df_use
    elif season == 'spring':
        for ax in axs:
            #  ax.set_ylim([0,35.0])
            ax.set_xlim([0,23])
        df_use = df_use.iloc[72:97,:]
        df_use.reset_index(drop=True, inplace=True)
        return df_use
    elif season == 'summer and winter':
        for ax in axs:
            #  ax.set_ylim([0,45.0])
            ax.set_xlim([0,23])
        df_use2 = df_use.iloc[48:72,:]
        df_use2.reset_index(drop=True, inplace=True)
        df_use = df_use.iloc[0:24,:]
        return df_use, df_use2

def cluster_summary_plots(n_clusters, radius, vertical=True, **kwargs): 
    """ 
    produces plots to compare the means of clusters using the DTW algorithm with
    different radii of wrapping window
    n_clusters (int): the number of clusters used in the DTW algorithm
    radii (list): list of radii to compare
    """

    #  cluster_file = 'rename_clusters_by_season.csv'
    cluster_file = f'../5_clusters_output/{n_clusters}_clusters_DTW_results_scaled_summer.csv'
    #  cluster_file = f'../RadiusComps/{n_clusters}_clusters_DTW_results_scaled_winter.csv'
    #  cluster_file = f'../RadiusComps/{n_clusters}_DTW_results_scaled_r{radius}.csv'
    df_cluster = pd.read_csv(cluster_file,
                             usecols=[1,2],
                            header=0,
                            index_col=0
                            )
    df_use1 = pd.read_pickle('../InputFiles/y1_SFR_hourly.pkl') 
    df_use2 = pd.read_pickle('../InputFiles/y2_SFR_hourly.pkl')
    df_use = pd.concat([df_use1, df_use2], join='inner')
    #  df_use1 = clean_outliers(df_use1)
    #  df_use2 = clean_outliers(df_use2)
    df_use = clean_outliers(df_use)
    df_use = df_use.multiply(7.48).round(2)

    if vertical:
        nrows = n_clusters
        ncols = 1
        figsize = (8, 40)
        sharex = True
        sharey = False
    else:
        nrows = 1
        ncols = n_clusters
        figsize = (48, 12)
        sharex = False
        sharey = True

    fig, axs = plt.subplots(nrows=nrows,
                            ncols=ncols,
                            figsize=figsize, 
                            sharex=sharex,
                            sharey=sharey,
                            layout='constrained')
    if kwargs:
        if 'period' in kwargs.keys():
            df_use = by_period(kwargs['period'], df_use, axs)

        if 'season' in kwargs.keys():
            df_use = by_season(kwargs['season'], df_use, axs)

    #  df_cluster.filter(items = [f'{season} Cluster'])
    clusters = list(range(n_clusters))
    fontsize = 18
    #  colors =
    cluster_names = ['Dominant Late Night',
                     'Steady Mid-Day',
                     'Dominant Early Morning',
                     'Dominant Evening',
                     'Dominant Morning']

    markers = ['o', 'v', '<', 's', 'D', '.']
    ylims = [40, 30, 55, 35, 45, 45]

    for c in clusters:
        cluster = [str(x) for x in df_cluster[df_cluster[f'summer cluster'] == c].index.to_list()]
        #  cluster = [str(x) for x in df_cluster[df_cluster[f'{season} Cluster'] == c].index.to_list()]
        df_use_c =  df_use.filter(items=cluster)
        #  df_use_cw = df_use2.filter(items=cluster)
        total = df_use_c.sum(axis=1)
        average = df_use_c.mean(axis=1)
        #  averagew = df_use_cw.mean(axis=1)
        peak = np.round(average.max(), 1)
        peakx = average.idxmax()
        #  for i in df_use_c.columns:
            #  i = str(i)
        #  axs[c].plot(df_use_c.index, df_use_c.iloc[:,278], c='grey')
        axs[c].plot(df_use_c.index, average, marker=markers[c], 
                    c=cluster_colors[c], linewidth=3, label='summer demand')
        #  if season == 'summer and winter':
            #  axs[c].plot(df_use_c.index, averagew, marker=markers[c],
                        #  linestyle='dotted', c=cluster_colors[c], linewidth=3,
                        #  label='winter demand')
            #  axs[c].legend(loc='upper left')
        axs[c].tick_params(axis='x', labelsize=16)
        axs[c].tick_params(axis='y', labelsize=16)
        axs[c].annotate(f'{cluster_names[c]}',
                        xy=(0.95, 0.95), xycoords='axes fraction',
                        ha='right', va='top',
                        fontsize=16
                        )
        #  axs[c].annotate(f'{peak}', xy=(peakx,peak), fontsize=14)
        #  axs[c].set_ylim([0, ylims[c]])
    fig.supxlabel('Time (hr)', fontsize=fontsize)
    fig.supylabel('Volume (gallons)', fontsize=fontsize)
    #  summer_line = mlines.Line2D([], [], color='black', label='Summer',
                                #  linewidth=3)
    #  winter_line = mlines.Line2D([], [], color='black', linestyle='dashed',
                                #  linewidth=3, label='Winter')
    #  fig.legend(handles=[summer_line, winter_line])
    #  fig.suptitle(f'{season}')
    plt.show()
    #  fig.savefig(f'../5_clusters_output/5_clusters_average_{season}_2.png')

def combine_season_clusters():
    year_clusters = pd.read_csv('../5_clusters_output/5_results_euclidean_dtw1.csv', 
                                header=0, 
                                usecols=[1,3], 
                                index_col=0)
    season_clusters = pd.read_csv('../5_clusters_output/rename_clusters_by_season.csv', 
                                    header=0,
                                    index_col=0)
    df_year = pd.DataFrame(year_clusters)
    df_season = pd.DataFrame(season_clusters)
    df = pd.concat([df_year, df_season], axis=1, join='inner')
    df.to_csv('../5_clusters_output/year_season_cluster.csv')
    return df

def prep_sankey():
    data = pd.read_csv('../5_clusters_output/year_season_cluster.csv', header=0,
                       index_col=0)
    df = pd.DataFrame(data)
    for n in df.iloc[:,0].unique():
        x0 = len(df[df.iloc[:,0] == n])
        breakpoint()
    #  x0 = df[:,0].value_counts()
        #  for i in df[df[]]
    #  for c, col in enumerate(df.columns):
        #  x0 = df[col].value_counts()
        #  for i in x0.index:
            #  print(i)
            #  breakpoint()
        #  x1 = []
        #  for i in range(5):
            #  x1 = df[df[col] == i].iloc[:,c+1].value_counts()
def plot_sankey_su_wi():

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node = dict(
          pad = 15,
          thickness = 20,
            align='left',
          line = dict(color = "black", width = 0.5),
          label = [
            "Total population", 
            "Summer DLN", "Summer SMD", "Summer DEM", "Summer DE","Summer DM",
            "Annual DLN", "Annual SMD", "Annual DEM", "Annual DE", "Annual DM",
            "Winter DLN", "Winter SMD", "Winter DEM", "Winter DE", "Winter DM"
          ],
            color = [
            "blue",
            'cornflowerblue', 'darkorange', 'forestgreen', 'tomato', 'mediumorchid',
            'cornflowerblue', 'darkorange', 'forestgreen', 'tomato', 'mediumorchid',
            'cornflowerblue', 'darkorange', 'forestgreen', 'tomato', 'mediumorchid'
            ]

        ),
        link = dict(
            source = [
                0, 0, 0, 0, 0, 
                1, 1, 1, 1, 1, 
                2, 2, 2, 2, 2, 
                3, 3, 3, 3, 3, 
                4, 4, 4, 4, 4, 
                5, 5, 5, 5, 5,
                #  6, 6, 6, 6, 6,
                #  7, 7, 7, 7, 7,
                #  8, 8, 8, 8, 8,
                #  9, 9, 9, 9, 9,
                #  10, 10, 10, 10, 10
            ], 
            target = [
                1, 2, 3, 4, 5, 
                #  6, 7, 8, 9, 10,
                #  6, 7, 8, 9, 10,
                #  6, 7, 8, 9, 10,
                #  6, 7, 8, 9, 10,
                #  6, 7, 8, 9, 10,
                11, 12, 13, 14, 15,
                11, 12, 13, 14, 15,
                11, 12, 13, 14, 15,
                11, 12, 13, 14, 15,
                11, 12, 13, 14, 15
            ],
            value = [
                1464, 4155, 3417, 5377, 3995,
                #1871, 4056, 2293, 5078, 5110, 
                #  1022, 75, 88, 600, 86, 
                #  62, 2828, 87, 394, 685, 
                #  229, 40, 1847, 104, 73,
                #  75, 897, 123, 3702, 281, 
                #  76, 315, 1272, 577, 2870,
                     900, 167, 113, 146, 138, 
                     155, 2761, 69, 747, 423, 
                     151, 297, 1978, 257, 734, 
                     718, 1057, 149, 2558, 895, 
                     140, 935, 186, 339, 2395]
      ))])

    fig.update_layout(font_size=18)
    fig.show()   
plot_sankey_su_wi()
def plot_sankey():

    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = ["Annual 0", "Summer 0", "Autumn 0", "Winter 0", "Spring 0",
                   "Annual 1", "Summer 1", "Autumn 1", "Winter 1", "Spring 1",
                   "Annual 2", "Summer 2", "Autumn 2", "Winter 2", "Spring 2",
                   "Annual 3", "Summer 3", "Autumn 3", "Winter 3", "Spring 3",
                   "Annual 4", "Summer 4", "Autumn 4", "Winter 4", "Spring 4",
                   ],
          color = "blue"
        ),
        link = dict(
          source = [0, 0, 0, 0, 0, 
                    5, 5, 5, 5, 5, 
                    10, 10, 10, 10, 10, 
                    15, 15, 15, 15, 15, 
                    20, 20, 20, 20, 20, 
                    1, 1, 1, 1, 1, 
                    6, 6, 6, 6, 6, 
                    11, 11, 11, 11, 11, 
                    16, 16, 16, 16, 16, 
                    21, 21, 21, 21, 21, 
                    2, 2, 2, 2, 2, 
                    7, 7, 7, 7, 7, 
                    12, 12, 12, 12, 12, 
                    17, 17, 17, 17, 17, 
                    22, 22, 22, 22, 22, 
                    3, 3, 3, 3, 3, 
                    8, 8, 8, 8, 8, 
                    13, 13, 13, 13, 13, 
                    18, 18, 18, 18, 18, 
                    23, 23, 23, 23, 23, 
                    4, 4, 4, 4, 4, 
                    9, 9, 9, 9, 9, 
                    14, 14, 14, 14, 14, 
                    19, 19, 19, 19, 19, 
                    24, 24, 24, 24, 24],
          target = [1, 6, 11, 16, 21, 
                    1, 6, 11, 16, 21,
                    1, 6, 11, 16, 21,
                    1, 6, 11, 16, 21,
                    1, 6, 11, 16, 21,
                    2, 7, 12, 17, 22, 
                    2, 7, 12, 17, 22, 
                    2, 7, 12, 17, 22, 
                    2, 7, 12, 17, 22, 
                    2, 7, 12, 17, 22, 
                    3, 8, 13, 18, 23,
                    3, 8, 13, 18, 23,
                    3, 8, 13, 18, 23,
                    3, 8, 13, 18, 23,
                    3, 8, 13, 18, 23,
                    4, 9, 14, 19, 24, 
                    4, 9, 14, 19, 24, 
                    4, 9, 14, 19, 24, 
                    4, 9, 14, 19, 24, 
                    4, 9, 14, 19, 24],
          #  value = [106, 2124, 40, 518, 1273,
                   #  78, 151, 998, 390, 3180,
                   #  502, 34, 805, 48, 57,
                   #  1308, 184, 42, 2523, 213,
                   #  114, 34, 805, 48, 57,
                   #  48, 395, 203, 141, 3274, 221, 110, 4054, 169, 234, 1275, 49,
                   #  53, 44, 25, 155, 491, 206, 3227, 191, 62, 49, 53, 44, 25, 60,
                   #  1298, 340, 262, 2101, 621, 353, 2810, 322, 691, 794, 149, 88,
                   #  320, 95, 66, 442, 420, 2811, 531, 55, 149, 88, 320, 95, 55,
                   #  2530, 208, 304, 964, 143, 281, 1919, 246, 2208, 1039, 118,
                   #  109, 111, 69, 252, 295, 464, 3095, 164, 66, 118, 109, 11, 69]
            #  value = [106,2124,40,518,1273,78,151,998,390,3180,502,34,805,48,57,1308,184,42,2523,213,114,1760,29,1860,84,
                    #  561,171,90,1154,132,561,171,90,1154,132,561,171,90,1154,132,561,171,90,1154,132,561,171,90,1154,132,
                    #  835,200,158,427,141,835,200,158,427,141,835,200,158,427,141,835,200,158,427,141,835,200,158,427,141,
                    #  693,121,539,113,130,693,121,539,113,130,693,121,539,113,130,693,121,539,113,130,693,121,539,113,130]
          value = [106,2124,40,518,1273,78,151,998,390,3180,502,34,805,48,57,1308,184,42,2523,213,114,1760,29,1860,84,
                    561,171,90,1154,132,85,1728,211,301,1928,873,66,856,62,57,105,2025,471,2056,682,137,156,2975,256,1283,
                    835,200,158,427,141,74,2458,247,779,588,552,372,2592,372,715,72,437,397,2424,499,63,1274,423,360,1962,
                    693,121,539,113,130,165,3120,235,1010,211,123,232,1273,320,1869,471,438,313,2947,193,103,1594,537,626,1045,]

      ))])

    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    fig.show()   
def add_parcel_size():
    lot = pd.read_csv(f'../5_clusters_output/cluster_lot_info.csv')
    lot_df = pd.DataFrame(lot)
    parcel = pd.read_csv('../InputFiles/parcel_size.csv', header=0,
                         usecols=[3,16]) 

    parcel_df = pd.DataFrame(parcel)
    #  print(parcel_df.head())
    df = lot_df.merge(parcel_df, how='left', left_on='Address', right_on='SitusAddre')
    df.to_csv('cluster_lot_info_parcel_size.csv')

def plot_boxplots(n_clusters, atts, anova=False):
    cluster_names = ['DLN',
                     'SMD',
                     'DEM',
                     'DE',
                     'DM']
    cols = atts + ['DBA cluster']
    #  atts.append('DBA cluster')
    lot_df = pd.read_csv(f'../5_clusters_output/cluster_lot_info_parcel_size.csv',
                usecols=cols)
    if 'SQFTmain' in atts:
        lot_df['SQFTmain'] = lot_df['SQFTmain'].replace(0, np.NaN)
        lot_df.dropna(subset=['SQFTmain'], inplace=True)
    if 'EffectiveYearBuilt' in atts:
        lot_df['EffectiveYearBuilt'] = 2018 - lot_df['EffectiveYearBuilt']
    #  lot_df.rename(columns={'EffectiveYearBuilt':'Home age',
                           #  'SQFTmain': 'House size (ft^2)',
                           #  'TotalValue': 'Home value ($)',
                           #  'Shape_Area': 'Lot size (ft^2)'},
                  #  inplace=True)
    
    #  atts = ['EffectiveYearBuilt',
            #  #  'Bedrooms',
            #  #  'Bathrooms',
            #  'SQFTmain',
            #  'TotalValue',
    #          'Shape_Area']
    att_labels = {'EffectiveYearBuilt': 'Home age',
                  'Bedrooms':'Bedrooms',
                  'Batchrooms': 'Bathrooms',
                  'SQFTmain': 'House size (ft^2)',
                  'TotalValue': 'Home value ($)',
                  'Shape_Area': 'Lot size (ft^2)'
                  }
    p_values = {}
    p_values.fromkeys(atts)
    #  fig, axs = plt.subplots(1, len(atts), tight_layout=True)
    fig, axs = plt.subplots(2, 2)
    axs = axs.flat
    for a, att in enumerate(atts):
        att_list = []
        for i in range(n_clusters):
            cluster_lot = lot_df.loc[lot_df['DBA cluster'] == i]
            att_list.append(cluster_lot[att])
        if anova:
            anova = stats.f_oneway(*att_list)
            p_values[att] = anova
        bp = axs[a].boxplot(att_list, 
                            labels=cluster_names, 
                            showmeans=True,
                            patch_artist=True,
                            boxprops={'linewidth':2.0},
                            whiskerprops={'linewidth':2.0},
                            medianprops={'linewidth':1.5,
                                         'color':'darkgray'},
                            capprops={'linewidth':2.0},
                            meanprops={'color':'black'}
                                 )
        for patch, color in zip(bp['boxes'], cluster_colors):
            patch.set_facecolor(color)
        axs[a].set_title(f'{att_labels[att]}', fontsize=16)
        axs[a].tick_params(labelsize=14)
        if att in ['TotalValue', 'Shape_Area']:
            axs[a].set_yscale('log')
    plt.show()
    if anova:
        print(p_values) 

def cluster_lot(n_clusters, radius):
    cluster_file = f'../RadiusComps/{n_clusters}_DTW_results_scaled_r{radius}.csv'
    cluster_data = pd.read_csv(cluster_file, 
                               usecols=[1,2], 
                               index_col=0)
    cluster_df = pd.DataFrame(cluster_data)
    cluster_df.index = cluster_df.index.map(str)
    census_df = pd.read_csv('../InputFiles/miu_census_tract_lot_2018.csv', 
                            usecols=[2,8,27], 
                            header=0)
    census_df = census_df[census_df['PropertyType'] == 'SFR']
    census_df.drop(columns=['PropertyType'], inplace=True)
    census_df.rename(columns={'MiuId': 'User'}, inplace=True)
    census_df.set_index('User', inplace=True)
    lot_df = pd.read_pickle('../InputFiles/lot_SFR.pkl') 
    lot_df.rename(columns={'MiuId': 'User'}, inplace=True)
    lot_df.set_index('User', inplace=True)
    lot_df.index = lot_df.index.map(str)
    lot_df = lot_df.join(cluster_df, how='inner')
    lot_df = lot_df.join(census_df, how='inner')
    lot_df.to_csv(f'{n_clusters}_cluster_r{radius}_lot_census_tract.csv')
    return lot_df

def cluster_census():
    file = pd.read_csv('../InputFiles/cluster_lot_census_tract.csv')
    df = pd.DataFrame(file)
    grouped = df.groupby(['TRACTCE', 'DBA cluster'])['DBA cluster'].count()
    grouped.to_csv('../5_clusters_output/cluster_by_censustract.csv')

def cluster_stacked_use(n_clusters, radius, period='year', operation='percent'):
    clusters = list(range(n_clusters))
    fontsize = 18

    fig, ax = plt.subplots()

    cluster_file = f'../RadiusComps/{n_clusters}_DTW_results_scaled_r{radius}.csv'
    cluster_data = pd.read_csv(cluster_file, 
                               usecols=[1,2],
                               index_col=0)
    cluster_df = pd.DataFrame(cluster_data)
    cluster_names = ['Dominant Late Night',
                     'Steady Mid-Day',
                     'Dominant Early Morning',
                     'Dominant Evening',
                     'Dominant Morning']

    df_use = pd.read_pickle('../InputFiles/y1_SFR_hourly.pkl')
    #  df_use2 = pd.read_pickle('../InputFiles/y2_SFR_hourly.pkl')
    #  df_use = pd.concat([df_use1, df_use2], join='inner')
    df_use = clean_outliers(df_use)
    if period == 'year':
        df_use = groupby_year(df_use)
        ax.set_xlim([0,23])
    elif period == 'month':
        df_use = groupby_month(df_use)
        ax.set_xlim([0,287])
    elif period == 'season':
        df_use = groupby_season(df_use)
        ax.set_xlim([0,95])
    else:
        print('keyword period must be one of "year", "month", or "season".')
    df_use = df_use.multiply(7.48)
    if operation == 'total':
        df_use = df_use.multiply(0.001)
    total_dict = {}
    average_dict = {}
    for c in clusters:
        cluster = [str(x) for x in cluster_df[cluster_df['DBA cluster'] == c].index.to_list()]
        df_cluster_use =  df_use.filter(items=cluster)
        total = df_cluster_use.sum(axis=1)
        average = df_cluster_use.mean(axis=1)
        total_dict[f'{cluster_names[c]}'] = total
        average_dict[f'{cluster_names[c]}'] = average
    total_total = sum(total_dict.values())
    average_total = sum(average_dict.values())
    percent_dict = {k: v / total_total for k, v in total_dict.items()} 

    if operation == 'total':
        y_values = total_dict
        y_label = 'Volume (gallons)'
    elif operation == 'percent':
        y_values = percent_dict
        y_label = ' '
        ax.set_ylim([0, 1.10])
        y_label = 'Volume (gallonsx10^3)'
    elif operation == 'average':
        y_values = average_dict
        y_label = 'Volume (gallons)'

    ax.stackplot(df_cluster_use.index, 
                  y_values.values(),
                  labels=total_dict.keys(),
                  colors=cluster_colors,
                  alpha=0.8)
    fig.legend(ncols=3,
               loc='upper center', 
               reverse=True, 
               fontsize=fontsize)
    ax.set_xlabel('Time (hr)', fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    #  fig3.savefig(f'{n_clusters}_{period}_clusters_stacked_total.png')
    plt.show()

def lot_cluster_hist(n_clusters):
    cluster_names = ['Dominant Late Night',
                     'Steady Mid-Day',
                     'Dominant Early Morning',
                     'Dominant Evening',
                     'Dominant Morning']
    lot_df = pd.read_csv(f'../5_clusters_output/cluster_lot_info_parcel_size.csv')
    lot_df['SQFTmain'] = lot_df['SQFTmain'].replace(0, np.NaN)
    lot_df.dropna(subset=['SQFTmain'], inplace=True)
    atts = [#'EffectiveYearBuilt', 
            'SQFTmain', 
            #  'Bedrooms',
            #  'Bathrooms',
            'TotalValue',
            #'Shape_Area'
    ]
    colors = ['red', 'yellowgreen', 'teal', 'lightsteelblue'] 
    fig, axs = plt.subplots(len(atts), n_clusters)#, tight_layout=True)#, sharey=True)
    n_bins = [21, 21, 21, 21]
    for a, att in enumerate(atts):
        #  fig, axs = plt.subplots(1, n_clusters, tight_layout=True, sharey=True)
        for i in range(n_clusters):
            xmin = lot_df[att].min()
            xmax = lot_df[att].max()
            cluster_lot = lot_df.loc[lot_df['DBA cluster'] == i]
            axs[a][i].hist(cluster_lot[att], 
                        bins=n_bins[a],
                        range=(xmin, xmax),
                        density=True, 
                        color = colors[a],
                        #  color=cluster_colors[i],
                        histtype='bar')
            #  axs[a][i].annotate(f'{cluster_names[i]}',
                        #  xy=(0.5, 0.95), xycoords='axes fraction',
                        #  ha='center', va='top',
                        #  fontsize=12
                        #  )
            axs[a][i].tick_params(labelsize=16)
            if a == 3:
                axs[a][i].set_xlabel(f'{cluster_names[i]}', fontsize=14)
            if i == 2:
                axs[a][i].set_title(f'{att}', fontsize=16)
            if i > 0:
                axs[a][i].sharey(axs[a][0])
                axs[a][i].tick_params(labelleft=False)
        #  fig.supxlabel(f'{att}')
    fig.supylabel('Probability density', fontsize=16)
    plt.show()

def cluster_lot(n_clusters, radius):
    cluster_file = f'../RadiusComps/{n_clusters}_DTW_results_scaled_r{radius}.csv'
    cluster_data = pd.read_csv(cluster_file, 
                               usecols=[1,2], 
                               index_col=0)
    cluster_df = pd.DataFrame(cluster_data)
    cluster_df.index = cluster_df.index.map(str)
    census_df = pd.read_csv('../InputFiles/miu_census_tract_lot_2018.csv', 
                            usecols=[2,8,27], 
                            header=0)
    census_df = census_df[census_df['PropertyType'] == 'SFR']
    census_df.drop(columns=['PropertyType'], inplace=True)
    census_df.rename(columns={'MiuId': 'User'}, inplace=True)
    census_df.set_index('User', inplace=True)
    lot_df = pd.read_pickle('../InputFiles/lot_SFR.pkl') 
    lot_df.rename(columns={'MiuId': 'User'}, inplace=True)
    lot_df.set_index('User', inplace=True)
    lot_df.index = lot_df.index.map(str)
    lot_df = lot_df.join(cluster_df, how='inner')
    lot_df = lot_df.join(census_df, how='inner')
    lot_df.to_csv(f'{n_clusters}_cluster_r{radius}_lot_census_tract.csv')
    return lot_df

def cluster_census():
    file = pd.read_csv('../InputFiles/cluster_lot_census_tract.csv')
    df = pd.DataFrame(file)
    grouped = df.groupby(['TRACTCE', 'DBA cluster'])['DBA cluster'].count()
    grouped.to_csv('../5_clusters_output/cluster_by_censustract.csv')

def census_trends():
    attrs = ['income', 'gt75', 'size', 'age']
    attr_dict = {'income': ['(a) median income ($)'],
                 'gt75': ['(b) % households with member >75 years old'],
                 'age': ['(d) median age of head-of-household'],
                 'size': ['(c) median number of household members']}
    #  attr_dict = {'income': ['(a)'],
                 #  'gt75': ['(b)'],
                 #  'age': ['(d)'],
                 #  'size': ['(c)']}

    r2_dict = {'income': {'DLN': 0.03, 
                          'SMD': [0.10, (79000, .21)],
                          'DEM': [0.29, (110000, .145)],
                          'DE': [0.13, (110000, .24)],
                          'DM': [0.18, (110000, .305)]
                          },
               'size': {'DLN': 0.00,
                        'SMD': 0.04,
                        'DEM': [0.20, (2.72, .115)],
                        'DE': [0.23, (2.71, .28)],
                        'DM': 0.00
                        },
               'gt75': {'DLN': [0.12, (7.6, .08)],
                        'SMD': [0.15, (3.3, .24)],
                        'DEM': [0.14, (7.9, .145)],
                        'DE': 0.095,
                        'DM': 0.083
                        },
               'age': {'DLN': [0.21, (35.5, .12)],
                        'SMD': 0.06,
                        'DEM': 0.01,
                        'DE': 0.01,
                        'DM': [0.19, (35.2, .27)]
                        }
               }

    cluster_dict = {'DLN': ['cornflowerblue', 'o'],
                       'SMD': ['darkorange', 'v'],
                       'DEM': ['forestgreen', '<'],
                       'DE': ['tomato', 's'],
                       'DM': ['mediumorchid', 'D']}

    data = pd.read_csv('../5_clusters_output/cluster_medians_by_census.csv',
                      header=0)
    df = pd.DataFrame(data)
    clusters = ['DLN', 'SMD', 'DEM', 'DE', 'DM']
    fig, axs = plt.subplots(2, 2, sharey=True)
    axs = axs.flat
    hand = []
    lab = []
    for a, att in enumerate(attrs):
        cl = []
        res = []
        for c in clusters:
            x = df[att]
            y = df[c]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            if r_value**2 > 0.088:
                cl.append(c) 
                res.append((intercept, slope))
        for i, c in enumerate(cl):
            x = df[att]
            y = df[c]
            axs[a].scatter(x,y, 
                           color=cluster_dict[c][0],
                           marker=cluster_dict[c][1],
                           label=c,
                           linewidth=2)
            axs[a].plot(x, res[i][0] + res[i][1]*x,
                        color=cluster_dict[c][0],
                        linewidth=2.5)
            axs[a].annotate(f'$R^2$={r2_dict[att][c][0]}', xy=r2_dict[att][c][1],
                            size=12)
            axs[a].set_xlabel(f'{attr_dict[att][0]}', fontsize=14)
            axs[a].tick_params(labelsize=14)
            #  axs[a].legend(loc='upper right', fontsize=12)
        if a in [0,2]:
            axs[a].set_ylabel('% population', fontsize=14)
        handles, labels = axs[a].get_legend_handles_labels()
        hand.append(handles)
        lab.append(labels)
    hand_flat = [x for xs in hand for x in xs]
    lab_flat = [x for xs in lab for x in xs]
    hand_lab = list(zip(hand_flat, lab_flat))
    unique_handles_labels = []
    seen_values = set()
    for i in hand_lab:
        if i[1] not in seen_values:
            seen_values.add(i[1])
            unique_handles_labels.append(i)
    handles_unique, labels_unique = list(zip(*unique_handles_labels))
    #  DEM_line = mlines.Line2D([], [], color='blue', marker='*',
                          #  markersize=15, label='Blue stars')
    fig.legend(handles_unique, 
               labels_unique, 
               loc='outside upper center',
               fontsize=14,
               ncols=len(labels_unique))
    plt.show()

def cluster_census_map():
    token = open(".public_token").read()
    #  file = '../5_clusters_output/cluster_lot_info.csv'
    file = '../5_clusters_output/cluster_by_censustract_percent.csv'
    cluster_lot = pd.read_csv(file, header=0) 
    #  cluster_lot['DBA cluster'] = cluster_lot['DBA cluster'].map(
    cluster_lot.rename(columns={'0': 'Dominant Late Night',
                                '1': 'Pronounced Early Morning',
                                '2': 'Dominant Early Morning',
                                '3': 'Dominant Evening',
                                '4': 'Dominant Morning'}, 
                       inplace=True)
    breakpoint()
    fig = px.scatter_mapbox(cluster_lot, 
                            lat="LAT",
                            lon="LON",
                            hover_name="TRACTCE",
                            zoom=20,
                            height=900)
    #  fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(mapbox_style="light", mapbox_accesstoken=token)
    fig.add_trace(go.Pie(values=cluster_lot.iloc[0,3:8], domain_x=(0.2,0.4),
                  domain_y=(0.1, 0.3)))
    #  fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

def cluster_census_pie():
    token = open(".public_token").read()
    file = '../5_clusters_output/cluster_by_censustract_percent.csv'
    cluster_lot = pd.read_csv(file, header=0)

    cluster_lot.rename(columns={'0': 'Dominant Late Night', 
                                '1': 'Pronounced Late Morning', 
                                '2': 'Dominant Early Morning', 
                                '3': 'Dominant Evening', 
                                '4': 'Dominant Morning'}, 
                       inplace=True)

    fig = go.Figure()
    segment_names = ['Dominant Late Night',
                     'Dominant Early Morning',
                     'Pronounced Late Morning',
                     'Dominant Evening',
                     'Dominant Morning']
    segment_colours = ['lightsteelblue',
                      'royalblue',
                      'mediumslateblue',
                      'mediumblue',
                      'midnightblue']
    #  segment_colours = ['midnightblue',
                       #  'mediumblue',
                       #  'mediumslateblue',
                       #  'royalblue',
                       #  'lightsteelblue']


    # Plot all segments
    for name, colour in zip(
        segment_names, segment_colours
    ):
        segment_trace = go.Scattermapbox(
            lon=cluster_lot["LON"],
            lat=cluster_lot["LAT"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=cluster_lot[name] * 200,
                sizemode="diameter",
                color=colour,
                opacity=0.8
            ),
            name=name,
            hoverinfo="text",
            text=cluster_lot['TRACTCE'],
            # ^ first characters of UK post code
        )
        fig.add_trace(segment_trace)

    # Update figure
    fig.update_layout(
        mapbox_style="light",
        mapbox_accesstoken=token,
        title_text="An interesting chart with bubbles & segments",
        width=800,
        height=600,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        margin=dict(b=0, t=0, l=0, r=0),
        legend=dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02),
    )
    fig.show()

def plot_cluster_map():
    #  token = open(".mapbox_token").read()
    token = open(".public_token").read()
    file = '../5_clusters_output/cluster_lot_info.csv'
    #  file = '9_cluster_r1_lot_census_tract.csv'
    cluster_lot = pd.read_csv(file, header=0,
                              usecols=['User', 'Bedrooms', 'TotalValue',
                                       'CENTER_LAT', 'CENTER_LON', 'DBA cluster'], 
                              dtype={'User':'string', 'Bedrooms':'Int32',
                                     'TotalValue':'Int32',
                                     'CENTER_LAT':'Float32',
                                     'CENTER_LON':'Float32', 'DBA cluster':'string'})
    cluster_lot['DBA cluster'] = cluster_lot['DBA cluster'].map({'0': 'Dominant Late Night',
                                                                '1': 'Pronounced Early Morning',
                                                                '2': 'Dominant Early Morning',
                                                                '3': 'Dominant Evening',
                                                                '4': 'Dominant Morning'})

    fig = px.scatter_mapbox(cluster_lot, 
                            lat="CENTER_LAT", 
                            lon="CENTER_LON",
                            labels={'DBA cluster': 'Cluster'},
                            hover_name="User", 
                            color="DBA cluster", 
                            color_discrete_sequence=cluster_colors,
                            zoom=10,
                            height=800)
    #  fig.update_layout(mapbox_style="open-street-map")
    fig.update_traces(cluster=dict(enabled=True))
    fig.update_layout(mapbox_style="light", mapbox_accesstoken=token)
    #  fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

def compare_dba_results():
    dfs = []
    for i in range(10):
        data = pd.read_csv(f'5_clusters_DTW_results_scaled_{i}.csv',
                           usecols=[1,2], header=0, index_col=0)
        df = pd.DataFrame(data)
        dfs.append(df)
    df_concat = pd.concat(dfs, axis=1, join='inner')
    df_concat.to_csv('5_clusters_DTW_results_comps.csv')

# for elasticity regression:

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
   #  block1 = ([0.0] * 6) + ([2.89] * 6)
   blockdiff1 = ([0.0] * 6) + ([2.89] * 6)
   blockdiff2 = ([3.5] * 6) + ([0.61] * 6)
   DOS = datetime_index.to_series().diff()[1:].astype('int64') / (864 * 10**11)

   data = {
       'FC': FC,
       #  'block1': block1,
       'blockdiff1': blockdiff1,
       'blockdiff2': blockdiff2,
       'DOS': DOS
   }
   df = pd.DataFrame(data, index=datetime_index[1:])
   df.to_pickle('instruments.pkl')
   #  return df
    #

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

    clusters = pd.read_csv(
        '../RadiusComps/5_DTW_results_scaled_r1.csv',
        usecols=[
            'User',
            'DBA cluster'],
        index_col='User')
    df_clusters = pd.DataFrame(clusters)
    df_clusters.index = df_clusters.index.map(str)

    weather = pd.read_pickle('weather_data.pkl')
    df_weather = pd.DataFrame(weather)


    if os.path.exists('instruments.pkl'):
        inst = pd.read_pickle('instruments.pkl')
    else:
        instruments()
        inst = pd.read_pickle('instruments.pkl')
    df_inst = pd.DataFrame(inst)
    
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
                                        df_bill.iloc[:,i].apply(np.log).round(3),
                                         df_clusters],
                                        axis=1,
                                        join='inner')
                all_list[i]['period'] = str((i % 6) + 1)
                all_list[i].columns = ['logQ', 'logP', 'cluster', 'period']
        elif kwargs['price'] == 'lagged average':
            for i, d in enumerate(all_list):
                if i > 0:
                    all_list[i] = pd.concat([d,
                                            df_bill.iloc[:,i-1],
                                             df_clusters],
                                             axis=1,
                                            join='inner')
                    all_list[i].columns = ['Q', 'P', 'cluster']
                    #  all_list[i]['period'] = str((i % 6) + 1)
                    all_list[i]['period'] = str(i + 1) 
                    all_list[i]['P_ave'] = np.log(all_list[i]['P'] /
                                                  all_list[i]['Q']).round(3)
                    all_list[i]['Q'] = all_list[i]['Q'].apply(np.log).round(3)
                    all_list[i] = all_list[i].assign(**df_inst.iloc[i-1,:])
                    all_list[i]['DOS_t'] = df_inst.iloc[i,3]
                    all_list[i] = all_list[i].assign(**df_weather.iloc[i,:])

    q_all = pd.concat(all_list[1:], axis=0, join='inner')
    q_all = q_all.reset_index(names='user')
    q_all.replace([np.inf, -np.inf, 'T'], np.nan, inplace=True)
    q_all.dropna(axis=0, how='any', inplace=True, ignore_index=False)
    if sample:
        if not kwargs:
            n_sample = 300
        else:
            n_sample = kwargs['n_sample']
        x = rng.choice(pd.unique(q_all['user']), n_sample, replace=False)
        q_all = q_all.loc[q_all['user'].isin(x)]
        #  q_all.to_pickle(f'LAP_inst_reg_data_{n_sample}.pkl')
        q_all.to_csv(f'LAP_inst_reg_data_{n_sample}.csv')
    else:
        #  q_all.to_pickle('LAP_inst_reg_data.pkl')
        q_all.to_stata('LAP_inst_reg_data.dta')

def add_dummies(file='reg_data.pkl'):
    file = file
    data = pd.read_pickle(
        file
    )
    data = pd.get_dummies(
        data=data,
        columns=["user", "period", "cluster"],
        drop_first=True,
        dtype=int
    )
    data.to_pickle(f'{file[:-4]}_with_dummies.pkl')
    # data.to_stata(f'H:/ckarren/Clustering/clustering/{file[:-4]}_with_dummies.dta')
    data.to_stata(f'{file[:-4]}_with_dummie.dta')

def users():
    users = pd.read_pickle('../InputFiles/user_ids.pkl')
    users = [int(x) for x in users]
    print(np.min(users))

def load_reg_data(data_file='B:/LAP_inst_reg_data_with_dummies.pkl', sample=False, **kwargs):
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
        data_file = 'B:/LAP_inst_reg_data_with_dummies.pkl'
        return data_file

def print_columns():
    print('0 | 1 |2     |3  |4            |5           |6        |7     |8      |9    |10   |11   |12   |13      |14  |15    |16 |17')
    print('-------------------------------------------------------------------------------------------------------------------------')
    print('Q | P |P_ave |FC |blockdiff1   |blockdiff2  |DOS_t-1  |DOS_t |ppdays |gt90 |pmax |tave |tmin |totalpp |cdd |rhave |pp |ET ')
    print('\n')
    print('-9       |-8       |-7       |-6       |-5       |-4        |-3        |-2        |-1')
    print('-----------------------------------------------------------------------------------------------')
    print('period_2 |period_3 |period_4 |period_5 |period_6 |cluster_1 |cluster_2 |cluster_3 |cluster_4')

def prep_ols(data):
    y = np.asarray(data['Q'], dtype=np.float32)             #dependent
    X = np.asarray(data['P_ave'])
    #  X = np.asarray(data.iloc[:,2:], dtype=np.float32)         #endog
    X = sm.add_constant(X)
    return y, X

def prep_lm_2sls(data):
    y = data['Q'].astype('float32')                 #dependent
    x0 = data[['DOS_t', 'gt90', 'totalpp', 'tmin', 'ET']].astype('float32')
    x1 = data.iloc[:,18:-9].astype('float32')
    X1 = pd.concat([x0, x1], axis=1).astype('float32')  #endog
    X = data['P_ave'].astype('float32')          #exog
    inst = data[['blockdiff2', 'DOS']].astype('float32')    #instrument
    return y, X1, X, inst

def prep_sm_2sls(data):
    y = np.asarray(data['Q'], dtype=np.float32)             #dependent
    X0 = data['P_ave']                                          #endog
    X1 = data.iloc[:,15:-9]                                       #exog
    X2 = data.iloc[:,-4:]
    X = np.asarray(pd.concat([X0, X1, X2], axis=1))
    X = sm.add_constant(X)
    inst = data.iloc[:,3:4]                                    #instrument
    inst = np.asarray(pd.concat([inst, X1, X2], axis=1))
    inst = sm.add_constant(inst)
    return y, X, inst
