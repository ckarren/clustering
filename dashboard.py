from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
#  from utils import *
import utils as ut
#  from data import all_time, all_period
from datetime import timedelta
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

app = Dash(__name__)

input_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/InputFiles/')
output_path = str('~/OneDrive - North Carolina State University/Documents/Clustering+Elasticity/OutputFiles/')
#  df_lot= pd.read_pickle(input_path + 'lot.pkl')
#  users = pd.read_pickle(input_path + 'user_ids.pkl')
#  users.insert(0,'All users')
water_use = pd.read_pickle(input_path + 'y1_SFR_hourly.pkl')
water_use = ut.clean_outliers(water_use)
users = list(water_use.columns)
init_user = users[0]
clusters = list(range(5))
n_clusters = 4

app.layout = html.Div([
    html.H4('Explore Lakewood Water User Clusters'), 
    html.Div([
        html.Div([
                html.P('Water User Cluster'), 
                dcc.Dropdown(
                    id='cluster', 
                    options=clusters, 
                    value=clusters[-1],
                    clearable=False,
                )
        ], style={'width': '31%', 'display': 'inline-block'}),
        
        html.Div([
            html.P('Sample size'),
                dcc.Dropdown(
                    id='sample',
                    options= list(range(100,1001,100)),
                    placeholder='Select a sample size'
                )
        ], style={'width': '31%', 'float': 'right', 'display': 'inline-block'}),
        
        html.Div([
            html.P('Feature'), 
                dcc.Dropdown(
                    id='feature',
                    options= [
                    'Average weekday demand pattern per month', 
                    'Average weekday demand pattern per season',
                    'Average weekday demand pattern per year'], 
                    value= 'Average weekday demand pattern per month',
                    clearable=False,                   
                )
        ], style={'width': '31%', 'float': 'center', 'display': 'inline-block'})
    ]), 

    dcc.Graph(id='time-series-chart'), 

    # dcc.RangeSlider()


])
#  @app.callback(
    #  Output('')
#  )
@app.callback(
    Output('time-series-chart', 'figure'),
    Input('user', 'value'),
    Input('sample', 'value'),
    Input('feature', 'value')
)
def display_time_series(user, sample, feature):
    #  water_use = pd.read_pickle(input_path + 'hourly_use_SFR_y1.pkl')
    #  sample = int(sample)
    #  water_use = water_use.sample(sample, axis=1, random_state=1)
    if feature == 'Average weekday demand pattern per month':
        average_use = ut.groupby_month(water_use)#[0]
    elif feature == 'Average weekday demand pattern per season':
        average_use = ut.groupby_season(water_use)#[0]
    elif feature == 'Average weekday demand pattern per year':
        average_use = ut.groupby_year(water_use)#[0]
    fig = px.line(average_use, x=average_use.index, y=average_use[user])

    #  fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.10,
                    #  rangeselector=dict(
                        #  buttons=list([
                            #  dict(count=1, label='1d', step='day', stepmode='backward'),
                            #  dict(count=7, label='1w', step='day', stepmode='backward'),
                            #  dict(count=5, label='1ww', step='day', stepmode='backward'),
                            #  dict(count=30, label='1m', step='day', stepmode='backward'),
                            #  dict(step='all')
                        #  ])
                    #  )
    #  )
    

    fig.update_layout(height=650, 
                    xaxis_title='Time', 
                    yaxis_title='Water Use (cubic feet)')
    return fig 

app.run_server(debug=True)


