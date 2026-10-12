import os

import utils as ut

import config
from data import DataLoader


class FeatureSelector:
    def __init__(self, input_path=None, output_path=None):
        self.loader = DataLoader(input_path=input_path)
        self.output_path = os.path.expanduser(output_path or config.OUTPUT_PATH)

    def pickle_feature(self):
        for i in range(1, 7):
            for j in range(1, 3):
                filename = f'hourly_use_SFR_y{j}_p{i}.pkl'
                df = self.loader.load_pickle(filename)
                df = ut.clean_outliers(df, 0.03, 400)
                week_df = df.groupby([df.index.weekday, df.index.hour]).mean()
                output_file = os.path.join(self.output_path, f'PDH_SFR_Y{j}P{i}.pkl')
                print(f'Writing {output_file}')
                week_df.to_pickle(output_file)


if __name__ == '__main__':
    FeatureSelector().pickle_feature()

