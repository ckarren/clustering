"""Data loading and cleaning utilities shared across scripts."""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd

import config
import utils as ut


class DataLoader:
    def __init__(self, input_path: str | None = None):
        self.input_path = os.path.expanduser(input_path or config.INPUT_PATH)

    def _full_path(self, filename: str) -> str:
        return os.path.join(self.input_path, filename)

    def load_pickle(self, filename: str) -> pd.DataFrame:
        return pd.read_pickle(self._full_path(filename))

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self._full_path(filename), **kwargs)

    def load_water_use(self, filename: str = 'y1_SFR_hourly.pkl', clean: bool = True) -> pd.DataFrame:
        df = self.load_pickle(filename)
        if clean:
            return ut.clean_outliers(
                df,
                lb=config.OUTLIER_LOWER_BOUND,
                ub=config.OUTLIER_UPPER_BOUND,
                ll=config.OUTLIER_LOWER_LIMIT,
            )
        return df

    def load_water_use_years(self, filenames: Iterable[str], clean: bool = True) -> pd.DataFrame:
        frames = [self.load_pickle(name) for name in filenames]
        df = pd.concat(frames, join='inner')
        if clean:
            return ut.clean_outliers(
                df,
                lb=config.OUTLIER_LOWER_BOUND,
                ub=config.OUTLIER_UPPER_BOUND,
                ll=config.OUTLIER_LOWER_LIMIT,
            )
        return df
