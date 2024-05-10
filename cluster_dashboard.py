import pandas as pd 
import utils as ut
import pickle

input_path = str('~/OneDrive - North Carolina State University/Documents/DynamicPricing/InputFiles/')
#
#  file1 = 'hourly_use_SFR_y1_p1.pkl'
#  file2 = 'hourly_use_SFR_y1_p2.pkl'
#  file3 = 'hourly_use_SFR_y1_p3.pkl'
#  file4 = 'hourly_use_SFR_y1_p4.pkl'
#  file5 = 'hourly_use_SFR_y1_p5.pkl'
#  file6 = 'hourly_use_SFR_y1_p6.pkl'
#
#  df1 = pd.read_pickle(input_path + file1)
#  df2 = pd.read_pickle(input_path + file2)
#  df3 = pd.read_pickle(input_path + file3)
#  df4 = pd.read_pickle(input_path + file4)
#  df5 = pd.read_pickle(input_path + file5)
#  df6 = pd.read_pickle(input_path + file6)

#  df1 = ut.clean_outliers(df1)
#  df2 = ut.clean_outliers(df2)
#  df3 = ut.clean_outliers(df3)
#  df4 = ut.clean_outliers(df4)
#  df5 = ut.clean_outliers(df5)
#  df6 = ut.clean_outliers(df6)

#  df = pd.concat([df1, df2, df3, df4, df5, df6])
#  df = df.fillna(0)
#  df = ut.clean_outliers(df)
#  user_list = list(df.columns)
with open('user_ids.pkl', 'rb') as f:
    user_list = pickle.load(f)
print(len(user_list))
#  print(len(users_list))
#  user = '1548226824'
#  test = check.resample('B').mean()
#  monday_means = (df.loc[(df.index.weekday == 0) &
                       #  (df.index.time == pd.to_datetime('01:00:00').time())]
                #  .mean())
                #  .to_frame('Monday 1 Am'))
#  monday_df = df.groupby(df.index.weekday).mean()
#  weekday_df = df.groupby(pd.Grouper(freq='B')).mean()
#  print(weekday_df)
#  week_df = df.groupby([df.index.month, df.index.weekday, df.index.hour]).mean()
#  print (week_df)
#  tuesday_means = (df.loc[(df.index.weekday == 1) &
                       #  (df.index.time == pd.to_datetime('01:00:00').time())]
                #  .mean())

