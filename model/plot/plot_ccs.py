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

from plot_utils import cumulate_data

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': 'mathpazo'
})

# =============================================================================
# DIRECTORIES
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.join(_cur_dir, '..')
_params_dir = os.path.join(_model_dir, 'params.xlsx')
_fig_dir = os.path.join(_model_dir, 'figures')
_res_dir = os.path.join(_model_dir, 'results')

tech_df = pd.read_excel(_params_dir, sheet_name='TECHNOLOGY', index_col=0,
    header=0)

fig, ax = plt.subplots(figsize=(4,3))

years = np.arange(2021,2041)
ccs = np.zeros(11)

#for scenario in ['ggpos_h2pos', 'ggpos_h2obl', 'ggobl_h2pos', 'ggobl_h2obl']:

for mix in range(11):

    scenario = 'ggobl_h2pos'
    scenario_mix = 'mix' + str(mix) + '_' + scenario

    _results_dir = os.path.join(_res_dir, scenario_mix)

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

            abated.loc[y,p] = sum(v_abated.loc[y,t] for t in tech_df.index 
                if tech_df.loc[t,'pillar'] == pillar) / 1e6

    ccs[mix] = abated['CCS'].sum()
    
mix = np.arange(0,11)
ax.plot(mix, ccs, color='#588157', linewidth=2.5)

ax.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)

ax.set_ylabel('Captured carbon emissions in Mt', fontsize=10)
ax.tick_params(axis='y', labelsize=8)
ax.set_ylim([ccs.min() - 10, ccs.max() + 10])

elec_mix = np.linspace(158, 0, num=len(mix)).astype(int)

ax.set_xlabel('Carbon intensity of the power mix in 2030 in $\mathrm{g/kWh}$',
    fontsize=10)
ax.set_xticks(mix)
ax.set_xticklabels(elec_mix, fontsize=8)

fig_name = '/ccs_' + scenario + '.png'
fig.savefig(_fig_dir + fig_name, dpi=1000)

plt.tight_layout()
plt.show()