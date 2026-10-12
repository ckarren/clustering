"""Reusable clustering orchestration for time-series data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

import config


@dataclass
class ClusteringConfig:
    n_clusters: int
    metric: str = 'dtw'
    random_state: int = config.DEFAULT_SEED
    n_init: int = config.DEFAULT_N_INIT
    max_iter_barycenter: int = config.DEFAULT_MAX_ITER_BARYCENTER
    verbose: bool = True
    scale: bool = True
    metric_params: dict[str, Any] = field(default_factory=dict)
    n_jobs: int | None = None

    @staticmethod
    def dtw(n_clusters: int, radius: int = config.DEFAULT_DTW_RADIUS, **kwargs) -> 'ClusteringConfig':
        metric_params = {
            'global_constraint': 'sakoe_chiba',
            'sakoe_chiba_radius': radius,
        }
        metric_params.update(kwargs.pop('metric_params', {}))
        return ClusteringConfig(
            n_clusters=n_clusters,
            metric='dtw',
            metric_params=metric_params,
            **kwargs,
        )

    @staticmethod
    def euclidean(n_clusters: int, **kwargs) -> 'ClusteringConfig':
        return ClusteringConfig(n_clusters=n_clusters, metric='euclidean', **kwargs)

    @staticmethod
    def softdtw(n_clusters: int, gamma: float = 0.01, **kwargs) -> 'ClusteringConfig':
        metric_params = {'gamma': gamma}
        metric_params.update(kwargs.pop('metric_params', {}))
        return ClusteringConfig(n_clusters=n_clusters, metric='softdtw', metric_params=metric_params, **kwargs)


class TimeSeriesClusterer:
    def __init__(self, cfg: ClusteringConfig):
        self.cfg = cfg

    def prepare(self, frame, scale: bool | None = None):
        dataset = to_time_series_dataset(frame)
        should_scale = self.cfg.scale if scale is None else scale
        if should_scale:
            dataset = TimeSeriesScalerMeanVariance().fit_transform(dataset)
        return dataset

    def fit_predict(self, frame):
        X_train = self.prepare(frame)
        kwargs: dict[str, Any] = {
            'n_clusters': self.cfg.n_clusters,
            'metric': self.cfg.metric,
            'verbose': self.cfg.verbose,
            'random_state': self.cfg.random_state,
            'n_init': self.cfg.n_init,
        }
        if self.cfg.metric == 'dtw':
            kwargs['max_iter_barycenter'] = self.cfg.max_iter_barycenter
        if self.cfg.metric_params:
            kwargs['metric_params'] = self.cfg.metric_params
        if self.cfg.n_jobs is not None:
            kwargs['n_jobs'] = self.cfg.n_jobs
        model = TimeSeriesKMeans(**kwargs)
        labels = model.fit_predict(X_train)
        return X_train, labels, model
