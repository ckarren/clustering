import glob
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
import pandas as pd 
import os

cluster_colors = ['cornflowerblue',
                  'darkorange',
                  'forestgreen',
                  'tomato',
                  'mediumorchid']

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
    df = df.T
    minmin = df.min().min()
    maxmax = df.max().max()
    clusters = df.iloc[:-1,:]
    total = df.iloc[-1:,:]
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    #  ax2.pcolormesh(clusters)
    im1 = ax1.imshow(clusters, 
                     vmin=minmin,
                     vmax=maxmax)
    #  ax2.xaxis.set_ticklabels([])
    #  ax2.yaxis.set_ticklabels(['1', '2', '3', '4', '5'])
    ax1.set_yticks(np.arange(len(df.index)-1), ['1', '2', '3', '4', '5'])
    ax1.set_ylabel('Cluster')
    #  ax1.pcolormesh(total)
    im2 = ax2.imshow()
    ax2.yaxis.set_ticklabels([])
    ax2.set_xlabel('Time (hr)')
    ax2.set_ylabel('Total')
    pad = 0.3

    #  divider = VBoxDivider(
        #  fig, 111,
        #  horizontal=[Size.AxesX(ax1), Size.Scaled(1), Size.AxesX(ax2)],
        #  vertical=[Size.AxesY(ax1), Size.Fixed(pad), Size.AxesY(ax2)])
#
    #  ax1.set_axes_locator(divider.new_locator(0))
    #  ax2.set_axes_locator(divider.new_locator(2))
    #  fig.colorbar(im1, ax=ax1, location='bottom', label='Volume (gallons)')
    fig.colorbar(im2, ax=ax2, location='bottom', label='Volume (gallons)')
    plt.show()

def cluster_hourly_heatmap_all(n_clusters, period='year'):
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
    df_use1 = clean_outliers(df_use1)
    df_use2 = clean_outliers(df_use2)
    df_use = clean_outliers(df_use)
    df_use = df_use.multiply(7.48).round(2)

    clusters = list(range(n_clusters))
    fig, axs = plt.subplots(nrows=1, ncols=n_clusters)

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

    for c in clusters:
        cluster = [str(x) for x in df_cluster[df_cluster['DBA cluster'] == c].index.to_list()]
        df_use_c =  df_use.filter(items=cluster)
        #  total = df_use_c.sum(axis=1)
        #  average = df_use_c.mean(axis=1)
        #  for i in df_use_c.columns:
            #  i = str(i)
        axs[c].imshow(df_use_c.index, df_use_c.columns)
        axs[c].plot(df_use_c.index, average, "o-", c=cluster_colors[c], linewidth=3)
        axs[c].tick_params(axis='x', labelsize=14)
        axs[c].tick_params(axis='y', labelsize=14)
        axs[c].annotate(f'{cluster_names[c]}', 
                        xy=(0.95, 0.95), xycoords='axes fraction', 
                        ha='right', va='top', 
                        fontsize=14
                        )
    fig.supxlabel('Time (hr)', fontsize=fontsize)
    fig.supylabel('Volume (gallons)', fontsize=fontsize)
    plt.show()

    #  ax2.pcolormesh(clusters)
    im1 = ax1.imshow(clusters, 
                     vmin=minmin,
                     vmax=maxmax)
    #  ax2.xaxis.set_ticklabels([])
    #  ax2.yaxis.set_ticklabels(['1', '2', '3', '4', '5'])
    ax1.set_yticks(np.arange(len(df.index)-1), ['1', '2', '3', '4', '5'])
    ax1.set_ylabel('Cluster')
    #  ax1.pcolormesh(total)
    im2 = ax2.imshow()
    ax2.yaxis.set_ticklabels([])
    ax2.set_xlabel('Time (hr)')
    ax2.set_ylabel('Total')
    pad = 0.3

    #  divider = VBoxDivider(
        #  fig, 111,
        #  horizontal=[Size.AxesX(ax1), Size.Scaled(1), Size.AxesX(ax2)],
        #  vertical=[Size.AxesY(ax1), Size.Fixed(pad), Size.AxesY(ax2)])
#
    #  ax1.set_axes_locator(divider.new_locator(0))
    #  ax2.set_axes_locator(divider.new_locator(2))
    #  fig.colorbar(im1, ax=ax1, location='bottom', label='Volume (gallons)')
    fig.colorbar(im2, ax=ax2, location='bottom', label='Volume (gallons)')
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
                     'Pronounced Late Morning',
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

def cluster_summary_plots(n_clusters, radius, vertical=True, **kwargs): 
    """ 
    produces plots to compare the means of clusters using the DTW algorithm with
    different radii of wrapping window
    n_clusters (int): the number of clusters used in the DTW algorithm
    radii (list): list of radii to compare
    """

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
    df_use1 = clean_outliers(df_use1)
    df_use2 = clean_outliers(df_use2)
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
        if kwargs['period'] == 'year':
            df_use = groupby_year(df_use)
            for ax in axs:
                ax.set_xlim([0,23])
                #  ax.set_ylim([0,45.0])
        elif kwargs['period'] == 'month':
            df_use = groupby_month(df_use)
            for ax in axs:
                ax.set_xlim([0,287])
                #  ax.set_ylim([0,45.0])
        elif kwargs['period'] == 'season':
            df_use = groupby_season(df_use)
            for ax in axs:
                ax.set_xlim([0,95])
                #  ax.set_ylim([0,45.0])
    #  else:
        #  print('keyword period must be one of "year", "month", or "season".')
    clusters = list(range(n_clusters))
    fontsize = 16
    #  colors =
    cluster_names = ['Dominant Late Night',
                     'Pronounced Late Morning',
                     'Dominant Early Morning',
                     'Dominant Evening',
                     'Dominant Morning']

    for c in clusters:
        cluster = [str(x) for x in df_cluster[df_cluster['DBA cluster'] == c].index.to_list()]
        df_use_c =  df_use.filter(items=cluster)
        total = df_use_c.sum(axis=1)
        average = df_use_c.mean(axis=1)
        #  for i in df_use_c.columns:
            #  i = str(i)
        #  axs[c].plot(df_use_c.index, df_use_c.iloc[:,278], c='grey')
        axs[c].plot(df_use_c.index, average, "o-", c=cluster_colors[c], linewidth=3)
        axs[c].tick_params(axis='x', labelsize=14)
        axs[c].tick_params(axis='y', labelsize=14)
        axs[c].annotate(f'{cluster_names[c]}', 
                        xy=(0.95, 0.95), xycoords='axes fraction', 
                        ha='right', va='top', 
                        fontsize=14
                        )
    fig.supxlabel('Time (hr)', fontsize=fontsize)
    fig.supylabel('Volume (gallons)', fontsize=fontsize)
    plt.show()

def lot_cluster_hist(n_clusters):
    cluster_names = ['Dominant Late Night',
                     'Pronounced Late Morning',
                     'Dominant Early Morning',
                     'Dominant Evening',
                     'Dominant Morning']
    lot_df = pd.read_csv(f'../5_clusters_output/cluster_lot_info.csv')
    lot_df['SQFTmain'] = lot_df['SQFTmain'].replace(0, np.NaN)
    lot_df.dropna(subset=['SQFTmain'], inplace=True)
    atts = ['EffectiveYearBuilt', 
            'SQFTmain', 
            'Bedrooms', 
            'Bathrooms', 
            'TotalValue']
    fig, axs = plt.subplots(len(atts), n_clusters)#, tight_layout=True)#, sharey=True)
    n_bins = [21,21,8,7,21]
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
                        color=cluster_colors[i],
                        histtype='bar')
            #  axs[a][i].annotate(f'{cluster_names[i]}',
                        #  xy=(0.5, 0.95), xycoords='axes fraction',
                        #  ha='center', va='top',
                        #  fontsize=12
                        #  )
       #  axs[a][i].set_title(f'{cluster_names[i]}')
            if a == 0:
                axs[a][i].set_title(f'{cluster_names[i]}', fontsize=14)
            if i == 2:
                axs[a][i].set_xlabel(f'{att}', fontsize=14)
            if i > 0:
                axs[a][i].sharey(axs[a][0])
        #  fig.supxlabel(f'{att}')
    fig.supylabel('Probability density')
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

def cluster_stacked_use(n_clusters, radius, period='year', operation='percent'):
    clusters = list(range(n_clusters))
    fontsize = 16

    fig3, ax3 = plt.subplots()

    cluster_file = f'../RadiusComps/{n_clusters}_DTW_results_scaled_r{radius}.csv'
    cluster_data = pd.read_csv(cluster_file, 
                               usecols=[1,2],
                               index_col=0)
    cluster_df = pd.DataFrame(cluster_data)
    cluster_names = ['Dominant Late Night',
                     'Pronounced Late Morning',
                     'Dominant Early Morning',
                     'Dominant Evening',
                     'Dominant Morning']

    df_use1 = pd.read_pickle('../InputFiles/y1_SFR_hourly.pkl')
    df_use2 = pd.read_pickle('../InputFiles/y2_SFR_hourly.pkl')
    df_use = pd.concat([df_use1, df_use2], join='inner')
    df_use = clean_outliers(df_use)
    if period == 'year':
        df_use = groupby_year(df_use)
        ax3.set_xlim([0,23])
    elif period == 'month':
        df_use = groupby_month(df_use)
        ax3.set_xlim([0,287])
    elif period == 'season':
        df_use = groupby_season(df_use)
        ax3.set_xlim([0,95])
    else:
        print('keyword period must be one of "year", "month", or "season".')
    df_use = df_use.multiply(7.48)
    total_dict = {}
    mean_dict = {}
    for c in clusters:
        cluster = [str(x) for x in cluster_df[cluster_df['DBA cluster'] == c].index.to_list()]
        df_cluster_use =  df_use.filter(items=cluster)
        total = df_cluster_use.sum(axis=1)
        average = df_cluster_use.mean(axis=1)
        total_dict[f'{cluster_names[c]}'] = total
    total_total = sum(total_dict.values())
    percent_dict = {k: v / total_total for k, v in total_dict.items()} 

    if operation == 'total':
        y_values = total_dict
        y_label = 'Volume (gallons)'
    elif operation == 'percent':
        y_values = percent_dict
        y_label = ' '
        ax3.set_ylim([0, 1.10])
        ax3.grid(which='both', axis='x')

    ax3.stackplot(df_cluster_use.index, 
                  y_values.values(),
                  labels=total_dict.keys(),
                  colors=cluster_colors,
                  alpha=0.8)
    ax3.legend(loc='upper right', reverse=True, fontsize=fontsize)
    ax3.set_xlabel('Time (hr)', fontsize=fontsize)
    ax3.set_ylabel(y_label, fontsize=fontsize)
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    #  fig3.savefig(f'{n_clusters}_{period}_clusters_stacked_total.png')
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
