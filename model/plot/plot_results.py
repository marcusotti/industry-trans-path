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
    'dir': _data_dir + '\DGG.xlsx'}
DGG_obl = {'name': 'DGG_obl',
    'dir': _data_dir + '\AT.xlsx'}
ELEC = {'name': 'ELEC',
    'dir': _data_dir + '\ELEC.xlsx'}

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

#ax.set_title('Energy demand of I&S, P&P and Cement Industry in Austria')

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

# =============================================================================
# PLOT CO2 EMISSIONS
from plot_utils import cumulate_data

# generate plot
fig_co2, ax_co2 = plt.subplots(figsize=(8, 5))

# read data
v_emissions = cumulate_data('v_emissions', scenario)
v_emissions.index = v_emissions.index.astype(str)
emissions = v_emissions.loc[years] /1e6

# plot emissions
ax_co2.bar(years, emissions, color='#606c38', label='CO2 emissions', hatch='//', edgecolor='white', alpha=alpha)

# CO2 abatement
tech_df = pd.read_excel(_params_dir, sheet_name='TECHNOLOGY', index_col=0, header=0)
v_abated = pd.read_excel(_results_dir + '\\v_abated.xlsx', sheet_name='data',
    index_col=0, header=0)

abated = pd.DataFrame(index=years, columns=['EEI', 'ELEC', 'FS', 'CCS'])

for y in years:
    for p in abated.columns:
        if p == 'EEI':
            pillar = 1
        elif p == 'ELEC':
            pillar = 2
        elif p == 'FS':
            pillar = 3
        else:
            pillar = 4

        abated.loc[y,p] = sum(v_abated.loc[int(y),t] for t in tech_df.index if tech_df.loc[t,'pillar'] == pillar) / 1e6

ax_co2.bar(years, abated['EEI'], color='#9381ff', label='EEI', bottom=emissions, alpha=alpha)
ax_co2.bar(years, abated['ELEC'], color='#ee9b00', label='ELEC', bottom=emissions + abated['EEI'], alpha=alpha)
ax_co2.bar(years, abated['FS'], color='#60d394', label='FS', bottom=emissions + abated['EEI'] + abated['ELEC'], alpha=alpha)
ax_co2.bar(years, abated['CCS'], color='#ee6055', label='CCS', bottom=emissions + abated['EEI'] + abated['ELEC'] + abated['FS'], alpha=alpha)

# add plot specs
ax_co2.legend(loc='upper center', facecolor='White', 
    fontsize=15, framealpha=0.8, handlelength=1, handletextpad=0.75, ncol=5, 
    borderpad=0.75, columnspacing=1, edgecolor="black", frameon=True)

ax_co2.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)
ax_co2.set_ylabel('CO2 emissions in Mt', fontsize=15)
ax_co2.tick_params(axis='y', labelsize=15)
ax_co2.set_ylim([0, emissions.max() * 1.2])
ax_co2.set_xticks(years)
ax_co2.set_xticklabels(years, fontsize=15)
ax_co2.tick_params(axis='x', length=0, pad=10)

fig_co2_name = '/co2_' + scenario['fig_name'] + '.png'
fig_co2.savefig(_fig_dir + fig_co2_name, dpi=1000)

# =============================================================================
# PLOT IMPORT RATIO
imp_ratio = {y: {} for y in years}

for y in years:
    data = pd.read_excel(_results_dir + '\\v_mix.xlsx', sheet_name=str(y),
        index_col=0, header=0)
    
    imp_ratio[y] = data.sum(axis=0)['grey'] / data.sum().sum()

imp_ratio_list = list(imp_ratio.values())
top_bars = [1 - val for val in imp_ratio_list]

# plot import ratio
fig_imp, ax_imp = plt.subplots(figsize=(8, 4))

ax_imp.bar(years, imp_ratio_list, color='#006699', edgecolor='white', hatch='//', alpha=alpha)
ax_imp.bar(years, top_bars, bottom=imp_ratio_list, color='#006699', alpha=alpha)

ax_imp.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)
ax_imp.tick_params(axis='x', length=0, pad=10, labelsize=15)
y_ticks = np.arange(0, 101, 20)
ax_imp.set_yticks(y_ticks/100)
ax_imp.set_yticklabels([f'{tick}\%' for tick in y_ticks], fontsize=15)
ax_imp.set_ylim([0, 1])

fig_imp_name = '/import_ratio_' + scenario['fig_name'] + '.png'
fig_imp.savefig(_fig_dir + fig_imp_name, dpi=1000)

# =============================================================================
# PLOT ENERGY DEMAND OF EEI SECTORS
# read data
sectors = [
    'Iron and Steel',
    'Pulp and Paper',
    'Non Metallic Minerals',
    'Chemical and Petrochemical',
    'Non Ferrous Metals',
    'Transport Equipment',
    'Machinery',
    'Mining and Quarrying',
    'Food Tobacco and Beverages',
    'Wood and Wood Products',
    'Construction',
    'Textiles and Leather',
    'Non Specified Industry'
]

demand = [
    (51482+31823+38329)/3600,
    73448/3600,
    39827/3600,
    44318/3600,
    9614/3600,
    6268/3600,
    23093/3600,
    6249/3600,
    25482/3600,
    26257/3600,
    16204/3600,
    2390/3600,
    7955/3600
]

dem_plot = [d / sum(demand) * 100 for d in demand]

tab20_colors = sns.color_palette("tab20", 20)  
dark_colors = tab20_colors[::2]  
light_colors = tab20_colors[1::2] 
custom_colors = dark_colors + light_colors 

sns.set_palette(custom_colors)
fig_eei, ax_eei = plt.subplots(figsize=(8, 5.5))

for i, sector in enumerate(sectors):
    label = sector
    if sector in ['Iron and Steel', 'Pulp and Paper', 'Non Metallic Minerals']:
        label = r'\textbf{' + sector + '}'
    ax_eei.bar(['2022', '2023'], [dem_plot[i], 0], width=0.5, alpha=alpha,
        bottom=sum(dem_plot[:i]), label=label)

handles, labels = ax_eei.get_legend_handles_labels()
legend = ax_eei.legend(handles[::-1], labels[::-1], loc='right', facecolor='White', bbox_to_anchor=(1, 0.5), 
                       fontsize=15, framealpha=0.8, handlelength=1, handletextpad=0.75, ncol=1, 
                       borderpad=0.75, columnspacing=1, edgecolor="black", frameon=True)

ax_eei.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)

ax_eei.set_xticks([])
ax_eei.set_xticklabels([])
ax_eei.set_yticklabels([f'{int(tick)}\%' for tick in ax_eei.get_yticks()], fontsize=15)

fig_eei.savefig(_fig_dir + "/sectors.png", dpi=1000)

# =============================================================================
# PLOT ENERGY CARRIER CONSUMPTION INDUSTRY
fig_carrier, ax_carrier = plt.subplots(figsize=(6, 4))

# read data
dem_carrier = {
    'Coal': (15478+51482+31823)/3600,
    'Oil and Oil Products': 17520/3600,
    'Natural Gas': 107118/3600,
    'Non Renewable Waste': 11964/3600,
    'Electricity': 99226/3600,
    'District Heat': 10259/3600,
    'Renewables and Biofuels': 57870/3600
}

colors_carrier = {
    'Coal': '#345053',
    'Oil and Oil Products': '#8b6e5a',
    'Natural Gas': '#a8b4b5',
    'Non Renewable Waste': '#ffba78',
    'Electricity': '#3e8bd8',
    'District Heat': '#f95738',
    'Renewables and Biofuels': '#36a35f',
}

carrier_labels = list(dem_carrier.keys())
carrier_demand = list(dem_carrier.values())
carrier_values = [d / sum(carrier_demand) * 100 for d in carrier_demand]

for i, label in enumerate(carrier_labels):
    ax_carrier.bar(['2022', '2023'], [carrier_values[i], 0], width=0.5, alpha=alpha,
        bottom=sum(carrier_values[:i]), label=label, color=colors_carrier[label])

handles, labels = ax_carrier.get_legend_handles_labels()
legend = ax_carrier.legend(handles[::-1], labels[::-1], loc='right', facecolor='White', bbox_to_anchor=(1, 0.5), 
                       fontsize=15, framealpha=0.8, handlelength=1, handletextpad=0.75, ncol=1, 
                       borderpad=0.75, columnspacing=1, edgecolor="black", frameon=True)

ax_carrier.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)

ax_carrier.set_xticks([])
ax_carrier.set_xticklabels([])
ax_carrier.set_yticklabels([f'{int(tick)}\%' for tick in ax_carrier.get_yticks()], fontsize=15)

fig_carrier.savefig(_fig_dir + "/industry_carriers.png", dpi=1000)

# =============================================================================
# PLOT ENERGY USE INDUSTRY
fig_use, ax_use = plt.subplots(figsize=(6, 4))

# read data
dem_use = {
    'IT and Lighting': 2,
    'Mechanical Energy': 23,
    r'LT Heat ($<150^\circ$C)': 23,
    r'MT Heat ($150-400^\circ$C)': 18,
    r'HT Heat ($>400^\circ$C)': 34
}

colors_use = {
    'IT and Lighting': '#3e8bd8',
    'Mechanical Energy': '#36a35f',
    r'LT Heat ($<150^\circ$C)': '#ffba08',
    r'MT Heat ($150-400^\circ$C)': '#e85d04',
    r'HT Heat ($>400^\circ$C)': '#9d0208'
}

use_labels = list(dem_use.keys())
use_demand = list(dem_use.values())
use_values = [d / sum(use_demand) * 100 for d in use_demand]

for i, label in enumerate(use_labels):
    ax_use.bar(['2022', '2023'], [use_values[i], 0], width=0.5, alpha=alpha,
        bottom=sum(use_values[:i]), label=label, color=colors_use[label])

handles, labels = ax_use.get_legend_handles_labels()
legend = ax_use.legend(handles[::-1], labels[::-1], loc='right', facecolor='White', bbox_to_anchor=(1, 0.5), 
                       fontsize=15, framealpha=0.8, handlelength=1, handletextpad=0.75, ncol=1, 
                       borderpad=0.75, columnspacing=1, edgecolor="black", frameon=True)

ax_use.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)

ax_use.set_xticks([])
ax_use.set_xticklabels([])
ax_use.set_yticklabels([f'{int(tick)}\%' for tick in ax_use.get_yticks()], fontsize=15)

fig_use.savefig(_fig_dir + "/industry_use.png", dpi=1000)

print('Carrier demand:')
print(carrier_demand)
print('Use demand:')
print(use_demand)


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

plt.tight_layout()
plt.show()
