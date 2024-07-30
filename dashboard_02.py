from dash import Dash, dcc, html, Input, Output
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import *
from data import all_time, all_period
from datetime import timedelta
import os 

app = Dash(__name__)

input_path = str('C:/Users/ckarren/OneDrive - North Carolina State University/Documents/DynamicPricing/InputFiles/')
output_path = str('C:/Users/ckarren/OneDrive - North Carolina State University/Documents/DynamicPricing/OutputFiles/')
df_lot= pd.read_pickle(input_path + 'lot.pkl') 

app.layout = html.Div([
    html.H4('Lakewood Water Use by Attribute and Parameter'), 
    html.Div([
        html.Div([
                html.P('Attribute'), 
                dcc.Dropdown(
                    id='attribute', 
                    options=['EffectiveYearBuilt', 'SQFTmain', 'Bedrooms', 'Bathrooms', 'TotalValue', 'TotalBill'], 
                    value='TotalValue', 
                    clearable=False,
                )
        ], style={'width': '31%', 'display': 'inline-block'}),
        
        html.Div([
            html.P('Year and Period'), 
                dcc.Dropdown(
                    id='period',
                    options= ['Y1P1 (Jul - Aug 2018)', 'Y1P2 (Sep - Oct 2018)',
                              'Y1P3 (Nov - Dec 2018)', 'Y1P4 (Jan - Feb 2019)',
                              'Y1P5 (Mar - Apr 2019)', 'Y1P6 (May - Jun 2019)', 
                              'Y2P1 (Jul - Aug 2019)', 'Y2P2 (Sep - Oct 2019)',
                              'Y2P3 (Nov - Dec 2019)', 'Y2P4 (Jan - Feb 2020)',
                              'Y2P5 (Mar - Apr 2020)', 'Y2P6 (May - Jun 2020)'],
                    value='Y1P1 (Jul - Aug 2018)',
                    clearable=False,
                )
        ], style={'width': '31%', 'float': 'right', 'display': 'inline-block'}),
        
        html.Div([
            html.P('Parameter'), 
                dcc.Dropdown(
                    id='quantile',
                    options= list(range(1,11)),
                    value='4',
                    clearable=False,                   
                )
        ], style={'width': '31%', 'float': 'center', 'display': 'inline-block'})
    ]), 

    dcc.Graph(id='time-series-chart'), 

    # dcc.RangeSlider()


])
#Filters: 'PropertyType', 'GeneralUseType', 'SpecificUseType', 
# SpecificUseDetail1, SpecificUseDetail2, YearBuilt, 
# EffectiveYearBuilt, SQFTmain, Bedrooms, Bathrooms, Units, LandValue, 
# TotalLandImpValue, TotalValue, 

# @app.callback(
#     Output('period', 'options'),
#     Input('year', 'value')
# )
# def set_month_options(selected_year):
#     return [{'label': i, 'value':i} for i in all_period[selected_year]]

# @app.callback(
#     Output('period', 'value'),
#     Input('period', 'options')
# )
# def set_month_value(available_options):
#     return available_options[0]['value']

@app.callback(
    Output('time-series-chart', 'figure'),
    Input('attribute', 'value'),
    #  Input('year', 'value'),
    Input('period', 'value'), 
    Input('quantile', 'value')
)
def display_time_series(attribute, period, quantile):
    q = int(quantile)
    file_name = period[0:4] + '_hourly.pkl'
    df_use = pd.read_pickle(input_path + file_name)
    df_use = tot_col(df_use)
    total_hr_use = df_use['Total']

    peak = total_hr_use.groupby(total_hr_use.index.date).idxmax()

    filt = miu_desc_filter(df_lot, 'GeneralUseType', 'Residential')
    df_use = df_use.reindex(filt, axis=1)
    df_use = clean_outliers(df_use, 0.03, 400)

    df_use = avg_col(df_use)
    aver_hr_use = df_use.groupby(df_use.index.date).AverageAll.mean()
    av = add_time(aver_hr_use)

    if attribute == 'TotalBill':
        file_name = 'bill_all.pkl'
        dfattr = pd.read_pickle(input_path + file_name)
        attr = period[0:4]
    else:
        attr = attribute
        dfattr = df_lot
    val_qrtl = qFilter(attr, dfattr, q)
    df_list = dfFilter(val_qrtl, df_use)
    df_q = newDF(df_list)

    wkend1, wkend2 = weekends(df_use)

    fig = px.line(df_q, x=df_q.index, y=df_q.columns)
    fig.add_trace(go.Scatter(x=av.index, y=aver_hr_use.values))

    for i  in range(len(wkend1)):
        fig.add_vrect(x0=wkend1[i], x1=wkend2[i], line_width=0, fillcolor='green', opacity=0.2)
    
    for i in peak:
        fig.add_vrect(x0=i-timedelta(minutes=30), x1=i+timedelta(minutes=30), line_width=0, fillcolor='red', opacity=0.2)

    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.10,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1d', step='day', stepmode='backward'), 
                            dict(count=7, label='1w', step='day', stepmode='backward'),
                            dict(count=5, label='1ww', step='day', stepmode='backward'),
                            dict(count=30, label='1m', step='day', stepmode='backward'), 
                            dict(step='all')
                        ])
                    )
    )
    

    fig.update_layout(height=850, 
                    xaxis_title='Time', 
                    yaxis_title='Water Use (cfs)')
    return fig 

app.run_server(debug=True)


