# AMI Clustering Analysis (Refactored)

This repository contains refactored time-series clustering and analysis workflows for AMI water-use data.

The refactor introduced:
- Clear module boundaries (configuration, data access, clustering engine, visualization)
- Class-based script runners with consistent entry points
- Standardized output naming and output directory behavior
- Backward-compatible utility wrappers for existing data processing functions

## Refactored Architecture

Core modules:
- `config.py`: Central configuration and defaults
- `data.py`: Shared data loading and cleaning (`DataLoader`)
- `clustering_engine.py`: Shared clustering setup and execution (`ClusteringConfig`, `TimeSeriesClusterer`)
- `visualizer.py`: Shared plotting helpers (`ClusterPlotter`)
- `utils.py`: Domain utilities and wrappers (`WaterUseDataProcessor` + compatibility functions)

Script pattern:
1. Load input data through `DataLoader`
2. Transform features with `utils` processing functions
3. Run model(s) through `TimeSeriesClusterer`
4. Save results/plots to the configured output path

## Configuration

Configuration lives in `config.py`.

Environment variables (optional):
- `AMI_INPUT_PATH` (default: `../InputFiles/`)
- `AMI_OUTPUT_PATH` (default: `../5_clusters_output/`)
- `AMI_SEED` (default: `0`)
- `AMI_N_INIT` (default: `5`)
- `AMI_MAX_ITER_BARYCENTER` (default: `20`)
- `AMI_DTW_RADIUS` (default: `1`)
- `AMI_OUTLIER_LB` (default: `1.0`)
- `AMI_OUTLIER_UB` (default: `400.0`)
- `AMI_OUTLIER_LL` (default: `-10.0`)

## Main Runners

All runners use `if __name__ == '__main__':` entry points.

Core clustering/comparison:
- `cluster_example.py` (`ClusterExampleRunner`)
- `cluster_compare_metrics.py` (`ClusterCompareMetricsRunner`)
- `compare_dba_clusters.py` (`CompareDbaClustersRunner`)
- `compare_dtw_cluster.py` (`CompareDtwClusterRunner`)
- `compare_euclidean_cluster.py` (`CompareEuclideanClusterRunner`)
- `compare_softdtw_cluster.py` (`CompareSoftDtwClusterRunner`)

Support analysis:
- `dtw_by_season.py` (`DtwBySeasonRunner`)
- `dtw_cluster_analysis.py` (`DTWClusterAnalysisRunner`)
- `cluster_summary.py` (`ClusterSummaryRunner`)
- `users.py` (`UserSampler`)
- `silouette_comp.py` (`SilhouetteComparisonRunner`)
- `plot_hourly_avg.py` (`HourlyAveragePlotter`)
- `test_clean.py` (`CleanOutlierCheck`)

Dashboard:
- `dashboard.py` (`WaterUseDashboard`)

## Output Convention

Refactored scripts now write outputs into `AMI_OUTPUT_PATH` / `config.OUTPUT_PATH`.

Naming convention used in refactored runners:
- Prefix with script/domain name
- Include `k{n}` for cluster count when applicable
- Include season/group suffix when applicable

Examples:
- `cluster_example_labels_k4.csv`
- `cluster_compare_metrics_summary.csv`
- `compare_dtw_cluster_k4_summer.csv`
- `dtw_by_season_k4.csv`
- `dtw_cluster_analysis_means_k5.png`

## Typical Workflow

1. Set input/output paths (optional via environment variables).
2. Run clustering script(s), for example:
   - `python compare_dtw_cluster.py`
   - `python dtw_by_season.py`
3. Review generated CSV/PNG outputs in the configured output directory.
4. Use `dashboard.py` for interactive exploration.

## Dependencies

Primary Python libraries used by refactored code:
- pandas
- numpy
- matplotlib
- plotly
- dash
- tslearn
- statsmodels
- linearmodels
- scipy

Install dependencies in your active environment before running scripts.

## Notes

- Utility processing functions (`groupby_year`, `groupby_month`, `groupby_season`, `clean_outliers`, etc.) remain available through `utils.py`.
- Several legacy scripts remain in the repository for historical context and may still use older patterns.
- Refactored scripts are the recommended execution path for new work.
