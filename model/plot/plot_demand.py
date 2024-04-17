import pyomo.environ as py
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import pandas as pd
import numpy as np
import math
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': 'mathpazo'
})

# =============================================================================
# DIRECTORIES
_data_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.join(_cur_dir, '..')
_fig_dir = os.path.join(_model_dir, 'figures')

# different scenarios
DGG = {'name': 'DGG',
    'dir': _data_dir + '\DGG.xlsx',
    'fig_name': 'DGG'}
DGG_obl = {'name': 'DGG_obl',
    'dir': _data_dir + '\AT.xlsx',
    'fig_name': 'DGG_obl'}
ELEC = {'name': 'ELEC',
    'dir': _data_dir + '\ELEC.xlsx',
    'fig_name': 'ELEC'}

# active scenario
scenario = ELEC # enter active scenario here
_params_dir = scenario['dir']

_results_dir = os.path.join(os.path.dirname(__file__),os.pardir, 'results',
    scenario['name'])

alpha = 0.9

# =============================================================================
# PLOT DEMAND

# generate plot
fig, ax = plt.subplots(figsize=(8, 6))

years = ['2021', '2025', '2030', '2035', '2040']
carriers = ['coal', 'alt', 'NG', 'GG', 'H2', 'elec']

# add colors
colors = {
    'elec': '#2F58CD',
    'NG': '#D9CB50',
    'coal': '#343A40',
    'H2': '#93BFCF',
    'alt': '#905E96',
    'GG': '#6A994E'
}

# read data
var = 'v_demand'
_file = os.path.join(_results_dir, var + '.xlsx')
file = pd.ExcelFile(_file)

site_df = pd.read_excel(_params_dir, sheet_name='SITE', 
    index_col='site', header=0)
site_df = site_df.drop(columns=['comments'])

IS_dem_y = pd.DataFrame(index=site_df.index, 
    columns=['elec', 'NG', 'coal', 'H2', 'alt'])
PP_dem_y = pd.DataFrame(index=site_df.index, 
    columns=['elec', 'NG', 'coal', 'H2', 'alt'])
NMM_dem_y = pd.DataFrame(index=site_df.index, 
    columns=['elec', 'NG', 'coal', 'H2', 'alt'])
IS_dem = []
PP_dem = []
NMM_dem = []

for sheet in file.sheet_names:
    df = pd.read_excel(_file, sheet_name=sheet, index_col=0)
    for site in site_df.index:
        if site_df.loc[site, 'branch'] == 'IS':
            IS_dem_y.loc[site,:] = df.loc[site,:]
        elif site_df.loc[site, 'branch'] == 'PP':
            PP_dem_y.loc[site,:] = df.loc[site,:]
        else:
            NMM_dem_y.loc[site,:] = df.loc[site,:]
        
    IS_dem.append(IS_dem_y.sum(axis=0))
    PP_dem.append(PP_dem_y.sum(axis=0))
    NMM_dem.append(NMM_dem_y.sum(axis=0))

IS_dem = pd.concat(IS_dem, axis=1, keys=file.sheet_names)
PP_dem = pd.concat(PP_dem, axis=1, keys=file.sheet_names)
NMM_dem = pd.concat(NMM_dem, axis=1, keys=file.sheet_names)

IS_dem_plot = IS_dem.loc[:,years] / 1e6
PP_dem_plot = PP_dem.loc[:,years] / 1e6
NMM_dem_plot = NMM_dem.loc[:,years] / 1e6

# include green gases
IS = pd.DataFrame(index=carriers, columns=years)
PP = pd.DataFrame(index=carriers, columns=years)
NMM = pd.DataFrame(index=carriers, columns=years)

for y in years:
    for c in IS_dem_plot.index:
        if c == 'NG':
            if y == '2021':
                GG_ratio = 0
            elif y == '2025':
                GG_ratio = 0.15
            elif y == '2030':
                GG_ratio = 0.4
            elif y == '2035':
                GG_ratio = 0.7
            else:
                GG_ratio = 1
            IS.loc['GG',y] = GG_ratio * IS_dem_plot.loc[c,y]
            IS.loc[c,y] = (1 - GG_ratio) * IS_dem_plot.loc[c,y]
            PP.loc['GG',y] = GG_ratio * PP_dem_plot.loc[c,y]
            PP.loc[c,y] = (1 - GG_ratio) * PP_dem_plot.loc[c,y]
            NMM.loc['GG',y] = GG_ratio * NMM_dem_plot.loc[c,y]
            NMM.loc[c,y] = (1 - GG_ratio) * NMM_dem_plot.loc[c,y]
        else:
            IS.loc[c,y] = IS_dem_plot.loc[c,y]
            PP.loc[c,y] = PP_dem_plot.loc[c,y]
            NMM.loc[c,y] = NMM_dem_plot.loc[c,y]

IS_dem_plot = IS
PP_dem_plot = PP
NMM_dem_plot = NMM


# Plots
bar_width = 0.28
bar_positions = np.arange(len(years))
IS_dem_offset = bar_positions - 0.02
PP_dem_offset = bar_positions + bar_width
NMM_dem_offset = bar_positions + 2 * bar_width + 0.02

# Plot the bars
ax.bar(IS_dem_offset, IS_dem_plot.loc['coal'], width=bar_width,
    label='Coal', color=colors['coal'], alpha=alpha)
ax.bar(PP_dem_offset, PP_dem_plot.loc['coal'], width=bar_width,
    label='Coal', color=colors['coal'], alpha=alpha)
ax.bar(NMM_dem_offset, NMM_dem_plot.loc['coal'], width=bar_width,
    label='Coal', color=colors['coal'], alpha=alpha)

ax.bar(IS_dem_offset, IS_dem_plot.loc['alt'], width=bar_width, alpha=alpha,
    bottom=IS_dem_plot.loc['coal'], label='Other fuels', color=colors['alt'])
ax.bar(PP_dem_offset, PP_dem_plot.loc['alt'], width=bar_width, alpha=alpha,
    bottom=PP_dem_plot.loc['coal'], label='Other fuels', color=colors['alt'])
ax.bar(NMM_dem_offset, NMM_dem_plot.loc['alt'], width=bar_width, alpha=alpha,
    bottom=NMM_dem_plot.loc['coal'], label='Other fuels', color=colors['alt'])

ax.bar(IS_dem_offset, IS_dem_plot.loc['NG'], width=bar_width, alpha=alpha,
    bottom=IS_dem_plot.loc['coal'] + IS_dem_plot.loc['alt'], label='Natural Gas', color=colors['NG'])
ax.bar(PP_dem_offset, PP_dem_plot.loc['NG'], width=bar_width, alpha=alpha,
    bottom=PP_dem_plot.loc['coal'] + PP_dem_plot.loc['alt'], label='Natural Gas', color=colors['NG'])
ax.bar(NMM_dem_offset, NMM_dem_plot.loc['NG'], width=bar_width, alpha=alpha,
    bottom=NMM_dem_plot.loc['coal'] + NMM_dem_plot.loc['alt'], label='Natural Gas', color=colors['NG'])

ax.bar(IS_dem_offset, IS_dem_plot.loc['GG'], width=bar_width, alpha=alpha,
    bottom=IS_dem_plot.loc['coal'] + IS_dem_plot.loc['NG'] + IS_dem_plot.loc['alt'], label='Green gases', color=colors['GG'])
ax.bar(PP_dem_offset, PP_dem_plot.loc['GG'], width=bar_width, alpha=alpha,
    bottom=PP_dem_plot.loc['coal'] + PP_dem_plot.loc['NG'] + PP_dem_plot.loc['alt'], label='Green gases', color=colors['GG'])
ax.bar(NMM_dem_offset, NMM_dem_plot.loc['GG'], width=bar_width, alpha=alpha,
    bottom=NMM_dem_plot.loc['coal'] + NMM_dem_plot.loc['NG'] + NMM_dem_plot.loc['alt'], label='Green gases', color=colors['GG'])

ax.bar(IS_dem_offset, IS_dem_plot.loc['H2'], width=bar_width, alpha=alpha,
    bottom=IS_dem_plot.loc['coal'] + IS_dem_plot.loc['NG'] + IS_dem_plot.loc['alt'] + IS_dem_plot.loc['GG'], label='Hydrogen', color=colors['H2'])
ax.bar(PP_dem_offset, PP_dem_plot.loc['H2'], width=bar_width, alpha=alpha,
    bottom=PP_dem_plot.loc['coal'] + PP_dem_plot.loc['NG'] + PP_dem_plot.loc['alt'] + PP_dem_plot.loc['GG'], label='Hydrogen', color=colors['H2'])
ax.bar(NMM_dem_offset, NMM_dem_plot.loc['H2'], width=bar_width, alpha=alpha,
    bottom=NMM_dem_plot.loc['coal'] + NMM_dem_plot.loc['NG'] + NMM_dem_plot.loc['alt'] + NMM_dem_plot.loc['GG'], label='Hydrogen', color=colors['H2'])

ax.bar(IS_dem_offset, IS_dem_plot.loc['elec'], width=bar_width, alpha=alpha,
    bottom=IS_dem_plot.loc['coal'] + IS_dem_plot.loc['NG'] + IS_dem_plot.loc['alt'] + IS_dem_plot.loc['GG'] + IS_dem_plot.loc['H2'], label='Electricity', color=colors['elec'])
ax.bar(PP_dem_offset, PP_dem_plot.loc['elec'], width=bar_width, alpha=alpha,
    bottom=PP_dem_plot.loc['coal'] + PP_dem_plot.loc['NG'] + PP_dem_plot.loc['alt'] + PP_dem_plot.loc['GG'] + PP_dem_plot.loc['H2'], label='Electricity', color=colors['elec'])
ax.bar(NMM_dem_offset, NMM_dem_plot.loc['elec'], width=bar_width, alpha=alpha,
    bottom=NMM_dem_plot.loc['coal'] + NMM_dem_plot.loc['NG'] + NMM_dem_plot.loc['alt'] + NMM_dem_plot.loc['GG'] + NMM_dem_plot.loc['H2'], label='Electricity', color=colors['elec'])

# add patches for legend
_patches = []
_patches.append(mpatches.Patch(color=colors['coal'], label='Coal'))
_patches.append(mpatches.Patch(color=colors['alt'], label='Other Fuels'))
_patches.append(mpatches.Patch(color=colors['NG'], label='Natural Gas'))
_patches.append(mpatches.Patch(color=colors['GG'], label='Green Gases'))
_patches.append(mpatches.Patch(color=colors['H2'], label='Hydrogen'))
_patches.append(mpatches.Patch(color=colors['elec'], label='Electricity'))

# add specs for plot
ax.legend(handles=_patches, loc='upper center', facecolor='White', 
    fontsize=15, framealpha=0.8, handlelength=1, handletextpad=0.75, ncol=3, 
    borderpad=0.75, columnspacing=1, edgecolor="black", frameon=True)

ax.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)
ax.set_ylabel('Energy demand in TWh', fontsize=15)
ax.set_yticklabels(ax.get_yticks(), fontsize=15)

# Adjust the x-axis ticks and labels
ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels(years, va='top', fontsize=15)
ax.tick_params(axis='x', which='major', length=0, pad=20)

positions = []
for i in range(len(years)):
    positions.append(IS_dem_offset[i])
    positions.append(PP_dem_offset[i] + 1e-4)
    positions.append(NMM_dem_offset[i])

ax.set_xticks(positions, minor=True) 
ax.set_xticklabels(['IS', 'PP', 'CEM'] * len(years), minor=True,  
    fontsize=12) 
ax.tick_params(axis='x', which='minor', length=0)

ax.set_ylim([0, IS_dem_plot.sum(axis=0).max() + 6.5])

fig_dem_name = '/demand_' + scenario['fig_name'] + '.png'
fig.savefig(_fig_dir + fig_dem_name, dpi=1000)

"""
# =============================================================================
# PLOT DEMAND IS

# generate plot
fig, ax = plt.subplots(figsize=(8, 6))

years = ['2021', '2025', '2030', '2035', '2040']
carriers = ['coal', 'alt', 'NG', 'GG', 'H2', 'elec']

# add colors
colors = {
    'elec': '#2F58CD',
    'NG': '#D9CB50',
    'coal': '#343A40',
    'H2': '#93BFCF',
    'alt': '#905E96',
    'GG': '#6A994E'
}

# read data
var = 'v_demand'
_file = os.path.join(_results_dir, var + '.xlsx')
file = pd.ExcelFile(_file)

site_df = pd.read_excel(_params_dir, sheet_name='SITE', 
    index_col='site', header=0)
site_df = site_df.drop(columns=['comments'])

IS_dem_y = pd.DataFrame(index=site_df.index, 
    columns=['elec', 'NG', 'coal', 'H2', 'alt'])

IS_dem = []

for sheet in file.sheet_names:
    df = pd.read_excel(_file, sheet_name=sheet, index_col=0)
    for site in site_df.index:
        if site_df.loc[site, 'branch'] == 'IS':
            IS_dem_y.loc[site,:] = df.loc[site,:]
        else:
            continue
        
    IS_dem.append(IS_dem_y.sum(axis=0))

IS_dem = pd.concat(IS_dem, axis=1, keys=file.sheet_names)

IS_dem_plot = IS_dem.loc[:,years] / 1e6

# include green gases
IS = pd.DataFrame(index=carriers, columns=years)

for y in years:
    for c in IS_dem_plot.index:
        if c == 'NG':
            if y == '2021':
                GG_ratio = 0
            elif y == '2025':
                GG_ratio = 0.15
            elif y == '2030':
                GG_ratio = 0.4
            elif y == '2035':
                GG_ratio = 0.7
            else:
                GG_ratio = 1
            IS.loc['GG',y] = GG_ratio * IS_dem_plot.loc[c,y]
            IS.loc[c,y] = (1 - GG_ratio) * IS_dem_plot.loc[c,y]
        else:
            IS.loc[c,y] = IS_dem_plot.loc[c,y]

IS_dem_plot = IS

# Plot the bars
ax.bar(years, IS_dem_plot.loc['coal'],
    label='Coal', color=colors['coal'], alpha=alpha)

ax.bar(years, IS_dem_plot.loc['alt'], alpha=alpha,
    bottom=IS_dem_plot.loc['coal'], label='Other fuels', color=colors['alt'])

ax.bar(years, IS_dem_plot.loc['NG'], alpha=alpha,
    bottom=IS_dem_plot.loc['coal'] + IS_dem_plot.loc['alt'], label='Natural Gas', color=colors['NG'])

ax.bar(years, IS_dem_plot.loc['GG'], alpha=alpha,
    bottom=IS_dem_plot.loc['coal'] + IS_dem_plot.loc['NG'] + IS_dem_plot.loc['alt'], label='Green gases', color=colors['GG'])

ax.bar(years, IS_dem_plot.loc['H2'], alpha=alpha,
    bottom=IS_dem_plot.loc['coal'] + IS_dem_plot.loc['NG'] + IS_dem_plot.loc['alt'] + IS_dem_plot.loc['GG'], label='Hydrogen', color=colors['H2'])

ax.bar(years, IS_dem_plot.loc['elec'], alpha=alpha,
    bottom=IS_dem_plot.loc['coal'] + IS_dem_plot.loc['NG'] + IS_dem_plot.loc['alt'] + IS_dem_plot.loc['GG'] + IS_dem_plot.loc['H2'], label='Electricity', color=colors['elec'])

# add patches for legend
_patches = []
_patches.append(mpatches.Patch(color=colors['coal'], label='Coal'))
_patches.append(mpatches.Patch(color=colors['alt'], label='Other Fuels'))
_patches.append(mpatches.Patch(color=colors['NG'], label='Natural Gas'))
_patches.append(mpatches.Patch(color=colors['GG'], label='Green Gases'))
_patches.append(mpatches.Patch(color=colors['H2'], label='Hydrogen'))
_patches.append(mpatches.Patch(color=colors['elec'], label='Electricity'))

# add specs for plot
ax.legend(handles=_patches, loc='upper center', facecolor='White', 
    fontsize=15, framealpha=0.8, handlelength=1, handletextpad=0.75, ncol=3, 
    borderpad=0.75, columnspacing=1, edgecolor="black", frameon=True)

ax.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)
ax.set_ylabel('Energy demand in TWh', fontsize=15)
ax.set_yticklabels(ax.get_yticks(), fontsize=15)

ax.set_title('Projected energy demand of Iron \& Steel industry in Austria', fontsize=20)

# Adjust the x-axis ticks and labels
ax.set_xticklabels(years, fontsize=15)

ax.set_ylim([0, IS_dem_plot.sum(axis=0).max() + 6.5])
"""
plt.tight_layout()
plt.show()
