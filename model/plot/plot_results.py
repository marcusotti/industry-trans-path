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

alpha = 0.8

# =============================================================================
# DIRECTORIES
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.join(_cur_dir, '..')
_params_dir = os.path.join(_model_dir, 'params.xlsx')
_fig_dir = os.path.join(_model_dir, 'figures')

# different scenarios
ggpos_h2pos = {
    'gg': False,
    'h2': False,
    'name': 'ggpos_h2pos'
}
ggpos_h2obl = {
    'gg': False,
    'h2': True,
    'name': 'ggpos_h2obl'
} 
ggobl_h2pos = {
    'gg': True,
    'h2': False,
    'name': 'ggobl_h2pos'
}
ggobl_h2obl = {
    'gg': True,
    'h2': True,
    'name': 'ggobl_h2obl'
}

scenarios = [
    ggpos_h2pos,
    ggpos_h2obl,
    ggobl_h2pos,
    ggobl_h2obl
]

for number in range(1):
    # active scenario
    scenario = scenarios[number]
    
    _results_dir = os.path.join(_model_dir, 'results')
    _results_dir = os.path.join(_results_dir, scenario['name'])
    if not os.path.exists(_results_dir):
        os.makedirs(_results_dir)

    # =============================================================================
    # PLOT DEMAND PER BRANCH
    years = ['2021', '2025', '2030', '2035', '2040']
    years_all = [str(year) for year in range(2021, 2041)]
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

    # generate plot
    fig, ax = plt.subplots(figsize=(8, 6))

    IS_dem_y = pd.DataFrame(index=site_df.index, 
        columns=carriers)
    PP_dem_y = pd.DataFrame(index=site_df.index, 
        columns=carriers)
    NMM_dem_y = pd.DataFrame(index=site_df.index, 
        columns=carriers)
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
    ax.set_ylim([0, IS_dem_plot.sum(axis=0).max() + 6.5])
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

    fig_dem_name = '/demand_' + scenario['name'] + '.png'
    fig.savefig(_fig_dir + fig_dem_name, dpi=1000)
    """
    # =============================================================================
    # PLOT TOTAL DEMAND OF THE SECTORS
    fig_sec, ax_sec = plt.subplots(figsize=(10, 5))
    
    IS_total = IS_dem.sum(axis=0) / 1e6
    PP_total = PP_dem.sum(axis=0) / 1e6
    NMM_total = NMM_dem.sum(axis=0) / 1e6

    ax_sec.plot(years_all, IS_total, label='Iron and Steel', color='#2a9d8f',
        linewidth=2.5)
    ax_sec.plot(years_all, PP_total, label='Pulp and Paper', color='#588157',
        linewidth=2.5)
    ax_sec.plot(years_all, NMM_total, label='Cement', color='#f4a261',
        linewidth=2.5)

    ax_sec.legend(loc='upper center', facecolor='White', fontsize=15, 
        framealpha=0.8, handlelength=1, handletextpad=0.75, ncol=4, 
        borderpad=0.75, columnspacing=1, edgecolor="black", frameon=True)

    ax_sec.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)
    ax_sec.set_ylabel('Energy demand in TWh', fontsize=15)
    ax_sec.set_ylim([0, IS_total.max() + 8.5])
    ax_sec.set_yticklabels(ax_sec.get_yticks(), fontsize=15)

    ax_sec.set_xticks(years_all)
    ax_sec.set_xticklabels([year if year in years else '' for year in years_all],
        fontsize=15)

    fig_total_name = '/demsectors_' + scenario['name'] + '.png'
    fig_sec.savefig(_fig_dir + fig_total_name, dpi=1000)
    """
    # =============================================================================
    # PLOT TOTAL DEMAND
    fig_dem, ax_dem = plt.subplots(figsize=(8, 6))

    dem = pd.DataFrame(index=years_all, columns=carriers)

    for sheet in file.sheet_names:
        df = pd.read_excel(_file, sheet_name=sheet, index_col=0)
        dem.loc[sheet, :] = df.sum(axis=0) / 1e6

    dem = dem.fillna(0)
    colors_dem = [
        '#343A40',
        '#905E96',
        '#D9CB50',
        '#6A994E',
        '#93BFCF',
        '#2F58CD'
    ]
    ax_dem.stackplot(years_all, dem['coal'], dem['alt'], dem['NG'], dem['GG'],
        dem['H2'], dem['elec'], colors=colors_dem, alpha=alpha)

    ax_dem.legend(handles=_patches, loc='upper center', facecolor='White', 
        fontsize=15, framealpha=0.8, handlelength=1, handletextpad=0.75, ncol=3, 
        borderpad=0.75, columnspacing=1, edgecolor="black", frameon=True)

    ax_dem.grid(which="major", axis="y", color="#758D99", alpha=0.4, zorder=1)
    ax_dem.set_ylabel('Energy demand in TWh', fontsize=15)
    ax_dem.set_ylim([0, dem.sum(axis=1).max() + 12.5])
    ax_dem.set_yticklabels(ax_dem.get_yticks(), fontsize=15)
    
    ax_dem.set_xticks(range(len(years_all)))
    ax_dem.set_xticklabels([year if year in years else '' for year in years_all],
        fontsize=15)

    fig_demtotal_name = '/totaldemand_' + scenario['name'] + '.png'
    fig_dem.savefig(_fig_dir + fig_demtotal_name, dpi=1000)
    
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

    fig_co2_name = '/co2_' + scenario['name'] + '.png'
    fig_co2.savefig(_fig_dir + fig_co2_name, dpi=1000)

#plt.tight_layout()
#plt.show()