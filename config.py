"""Project-wide configuration constants for clustering workflows."""

import os

INPUT_PATH = os.getenv('AMI_INPUT_PATH', '../InputFiles/')
OUTPUT_PATH = os.getenv('AMI_OUTPUT_PATH', '../5_clusters_output/')

DEFAULT_SEED = int(os.getenv('AMI_SEED', '0'))
DEFAULT_N_INIT = int(os.getenv('AMI_N_INIT', '5'))
DEFAULT_MAX_ITER_BARYCENTER = int(os.getenv('AMI_MAX_ITER_BARYCENTER', '20'))
DEFAULT_DTW_RADIUS = int(os.getenv('AMI_DTW_RADIUS', '1'))

OUTLIER_LOWER_BOUND = float(os.getenv('AMI_OUTLIER_LB', '1.0'))
OUTLIER_UPPER_BOUND = float(os.getenv('AMI_OUTLIER_UB', '400.0'))
OUTLIER_LOWER_LIMIT = float(os.getenv('AMI_OUTLIER_LL', '-10.0'))

CLUSTER_COLORS = [
    'cornflowerblue',
    'darkorange',
    'forestgreen',
    'tomato',
    'mediumorchid',
]
