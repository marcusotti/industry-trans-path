import pyomo.environ as py
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import matplotlib.patches as patch
import pandas as pd
import numpy as np
import math
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from matplotlib import rcParams
from matplotlib.lines import Line2D

def plot_sankey(labels, source, target, values, carrier):
    colors = {
        'Coal': '#dad7cd',
        'Coke': '#343a40', 
        'Natural Gas': '#e9c46a',
        'Electricity': '#2a9d8f',
        'Manufacture Gas': '#dda15e',
        'Losses': '#caf0f8',
        'Heat': '#e76f51'
    }

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=10,
            line=dict(color='black', width=0.1),
            # label=labels,
            color='#edede9'
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            # customdata=carrier,
            color=[colors[i] for i in carrier]
        )
    )])

    legend_data = []
    for label, color in colors.items():
        if label in carrier:
            legend_data.append(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color=color, size=20, symbol='square'),
                name=label
            ))

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.add_traces(legend_data)
    fig.update_layout(
        showlegend=False, 
        plot_bgcolor='white' 
    )
    return fig

def cumulate_data(var, scenario):
    # read data from excel file
    _results_dir = os.path.join(os.path.dirname(__file__),os.pardir, 'results',
    scenario['name'])
    
    _file = os.path.join(_results_dir, var + '.xlsx')
    file = pd.ExcelFile(_file)

    num_sheets = len(file.sheet_names)
    if num_sheets == 1:
        df = pd.read_excel(_file, sheet_name='data', index_col=0)
        data = df.sum(axis=1)
    elif num_sheets > 1:
        data = []
        for sheet in file.sheet_names:
            df = pd.read_excel(_file, sheet_name=sheet, index_col=0)
            df_sum = df.sum(axis=0)
            data.append(df_sum)
        data = pd.concat(data, axis=1, keys=file.sheet_names)    

    return data