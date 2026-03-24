#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pandas as pd

import config
from data import DataLoader



class UserSampler:
    def __init__(self, n_sample=100, random_state=1):
        self.n_sample = n_sample
        self.random_state = random_state
        self.loader = DataLoader()
        self.output_path = os.path.expanduser(config.OUTPUT_PATH)
        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        use_df = self.loader.load_water_use('y1_SFR_hourly.pkl', clean=True)
        use_df = use_df.sample(n=self.n_sample, axis=1, random_state=self.random_state)
        user_df = pd.DataFrame(use_df.columns)
        output_file = os.path.join(self.output_path, f'user_sample_n{self.n_sample}.csv')
        user_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    UserSampler().run()
