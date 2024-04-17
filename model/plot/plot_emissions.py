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

years = ['2021', '2025', '2030', '2035', '2040']

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


plt.tight_layout()
plt.show()
