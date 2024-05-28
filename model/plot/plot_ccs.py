import pyomo.environ as py
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import pandas as pd
import numpy as np
import math
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
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

years = np.arange(2021,2041)
scenarios = ['ggpos_h2pos', 'ggpos_h2obl', 'ggobl_h2pos', 'ggobl_h2obl']
ccs = pd.DataFrame(index=scenarios, columns=np.arange(11))

for scenario in scenarios:
    for mix in range(11):
        scenario_mix = 'mix' + str(mix) + '_' + scenario

        _results_dir = os.path.join(_res_dir, scenario_mix)

        v_abated = pd.read_excel(_results_dir + '\\v_abated.xlsx',
            sheet_name='data', index_col=0, header=0)

        abated = pd.DataFrame(index=years, columns=['CCS'])

        for y in years:
            abated.loc[y,'CCS'] = sum(v_abated.loc[y,t] for t in tech_df.index 
                if tech_df.loc[t,'pillar'] == 4) / 1e6

        ccs.loc[scenario, mix] = abated['CCS'].sum()

mix = np.arange(0,11)

for scenario in scenarios:
    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.plot(mix, ccs.loc[scenario], color='#588157', linewidth=3.5)
    ax.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)
    ax.set_ylabel('Carbon emissions captured in Mt', fontsize=17)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_ylim([88, 139])
    ax.set_xlabel('Carbon intensity of the power mix in 2030 in $\mathrm{g/kWh}$',
        fontsize=17)
    ax.set_xticks(mix)
    elec_mix = np.linspace(160, 0, num=len(mix)).astype(int)
    ax.set_xticklabels(elec_mix, fontsize=15)

    def draw_circle(ax, x, y, radius, **kwargs):
        # Get the aspect ratio of the axes
        aspect_ratio = ax.get_data_ratio()
        width = radius * 2 / aspect_ratio
        height = radius * 2
        ellipse = Ellipse((x, y), width, height, **kwargs)
        ax.add_patch(ellipse)


    first_value = ccs.loc[scenario, 0]
    draw_circle(ax, 0, first_value, 0.7, edgecolor='black', facecolor='none', lw=2, zorder=2)
    ax.annotate(f'{first_value:.1f} Mt',
                xy=(0, first_value + 0.5), xycoords='data',
                xytext=(1, first_value + 6), textcoords='data',
                arrowprops=dict(arrowstyle="-",
                    connectionstyle='angle,angleA=0,angleB=90,rad=0'),
                fontsize=15, color='black')
    
    last_value = ccs.loc[scenario, 10]
    draw_circle(ax, 10, last_value, 0.7, edgecolor='black', facecolor='none', lw=2, zorder=2)
    ax.annotate(f'{last_value:.1f} Mt',
                xy=(10, last_value - 0.5), xycoords='data',
                xytext=(9, last_value - 6), textcoords='data',
                arrowprops=dict(arrowstyle="-", 
                    connectionstyle='angle,angleA=180,angleB=90,rad=0'),
                fontsize=15, color='black', horizontalalignment='right')
    
    plt.tight_layout()
    fig_name = '/ccs_' + scenario + '.pdf'
    fig.savefig(_fig_dir + fig_name, dpi=1000)

#plt.show()