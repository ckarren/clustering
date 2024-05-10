import pandas as pd 
import utils as ut 

input_path = str('~/OneDrive - North Carolina State University/Documents/DynamicPricing/InputFiles/')
output_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')

def pickle_feature(input_path, output_path):
    for i in range(1,7):
        for j in range(1,3):
            file = f'hourly_use_SFR_y{j}_p{i}.pkl'
            df = pd.read_pickle(input_path + file)
            df = ut.clean_outliers(df, 0.03, 400)
            week_df = df.groupby([df.index.weekday, df.index.hour]).mean()
            print(f'Writing PDH_SFR_Y{j}P{i}.pkl file')
            week_df.to_pickle(output_path + f'PDH_SFR_Y{j}P{i}.pkl')
    #  monday_means = (df.loc[(df.index.weekday == 0) &
                           #  (df.index.time == pd.to_datetime('01:00:00').time())]
                    #  .mean())
                    #  .to_frame('Monday 1 Am'))
pickle_feature(input_path, output_path)

