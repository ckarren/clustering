import warnings

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

import utils as ut
from data import DataLoader

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class WaterUseDashboard:
    FEATURE_MONTH = 'Average weekday demand pattern per month'
    FEATURE_SEASON = 'Average weekday demand pattern per season'
    FEATURE_YEAR = 'Average weekday demand pattern per year'

    def __init__(self, input_path='~/Documents/Clustering+Elasticity/InputFiles/'):
        self.input_path = str(input_path)
        self.loader = DataLoader(input_path=self.input_path)
        self.app = Dash(__name__)
        self.water_use = self._load_water_use()
        self.features = [
            self.FEATURE_MONTH,
            self.FEATURE_SEASON,
            self.FEATURE_YEAR,
        ]
        self.clusters = list(range(5))

        self.app.layout = self._build_layout()
        self._register_callbacks()

    def _load_water_use(self):
        return self.loader.load_water_use('y1_SFR_hourly.pkl', clean=True)

    def _build_layout(self):
        return html.Div([
            html.H4('Explore Lakewood Water User Clusters'),
            html.Div([
                html.Div([
                    html.P('Water User Cluster'),
                    dcc.Dropdown(
                        id='cluster',
                        options=self.clusters,
                        value=self.clusters[-1],
                        clearable=False,
                    ),
                ], style={'width': '31%', 'display': 'inline-block'}),
                html.Div([
                    html.P('Sample size'),
                    dcc.Dropdown(
                        id='sample',
                        options=list(range(100, 1001, 100)),
                        placeholder='Select a sample size',
                    ),
                ], style={'width': '31%', 'float': 'right', 'display': 'inline-block'}),
                html.Div([
                    html.P('Feature'),
                    dcc.Dropdown(
                        id='feature',
                        options=self.features,
                        value=self.FEATURE_MONTH,
                        clearable=False,
                    ),
                ], style={'width': '31%', 'float': 'center', 'display': 'inline-block'}),
            ]),
            dcc.Graph(id='time-series-chart'),
        ])

    def _resolve_feature_dataframe(self, feature):
        feature_map = {
            self.FEATURE_MONTH: ut.groupby_month,
            self.FEATURE_SEASON: ut.groupby_season,
            self.FEATURE_YEAR: ut.groupby_year,
        }
        transform = feature_map.get(feature, ut.groupby_month)
        return transform(self.water_use)

    def _build_time_series_figure(self, feature):
        average_use = self._resolve_feature_dataframe(feature)
        fig = px.line(average_use, x=average_use.index, y=average_use[0])
        fig.update_layout(
            height=650,
            xaxis_title='Time',
            yaxis_title='Water Use (cubic feet)',
        )
        return fig

    def _register_callbacks(self):
        @self.app.callback(
            Output('time-series-chart', 'figure'),
            Input('sample', 'value'),
            Input('feature', 'value'),
        )
        def display_time_series(sample, feature):
            _ = sample
            return self._build_time_series_figure(feature)

    def run(self, debug=True):
        self.app.run_server(debug=debug)


if __name__ == '__main__':
    WaterUseDashboard().run(debug=True)


