import pyomo.environ as py
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import matplotlib.patches as patch
import pandas as pd
import numpy as np
import math
import cloudpickle
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# initialize figure
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': 'mathpazo'
})

fig = plt.figure(figsize=(10, 6), constrained_layout=False)
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
plt.tight_layout()

# load data
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_shapefile_dir = os.path.join(_cur_dir, 'shapefile')
_model_dir = os.path.join(_cur_dir, '..')
_data_dir = os.path.join(_model_dir, 'data')
_fig_dir = os.path.join(_model_dir, 'figures')

# =============================================================================
# plot regions and gas grid
nuts_shp = gpd.read_file(_shapefile_dir + '/NUTS_RG_20M_2021_4326.shp')
lau_shp = gpd.read_file(_shapefile_dir + '/at_lau.shp')

# specify country and regions
AT = nuts_shp.loc[nuts_shp['CNTR_CODE']=='AT']
AT_0 = AT.loc[AT['LEVL_CODE']==0]
AT_2 = AT.loc[AT['LEVL_CODE']==2]
AT_3 = AT.loc[AT['LEVL_CODE']==3]

# plot
AT_0.boundary.plot(ax=ax, linewidth=2, color='black', zorder=3)
AT_2.boundary.plot(ax=ax, linewidth=1.5, color='black', zorder=2)
AT_3.boundary.plot(ax=ax, linewidth=1, color='#6c757d', zorder=1)
# lau_shp.boundary.plot(ax=ax, linewidth=0.2, color='grey')

for idx, row in AT_3.iterrows():
    centroid = row['geometry'].representative_point()
    ax.annotate(row['NUTS_ID'], xy=(centroid.x, centroid.y), fontsize=10,
    ha='center', color='#6c757d', alpha=0.8, fontweight='bold', zorder=4)

# =============================================================================
# include industrial sites in the plot
# identify location of industrial sites
sites = pd.read_excel(_model_dir + '/params.xlsx', sheet_name='SITE', 
        index_col='site', header=0)
sites = sites.drop(['comments', 'output', 'SEC', 'SCe', 'elec', 'NG', 'coal', 'H2', 'alt'], axis=1)

branch_color = {
        'Iron & Steel': '#118ab2',
        'Pulp & Paper': '#ef476f',
        'Cement': '#ffbd00'
        }

for s in sites.index:
    loc = sites.loc[s, 'location']
    if loc in lau_shp['LAU_NAME'].values:
        geo = lau_shp.loc[lau_shp['LAU_NAME'] == loc, 'geometry'].squeeze()
        if not geo.is_empty:
            center = geo.representative_point()
            branch = sites.loc[s, 'branch']
            color = (
                branch_color['Iron & Steel'] if branch == 'IS' 
                else branch_color['Pulp & Paper'] if branch == 'PP' 
                else branch_color['Cement']
                )
            ax.plot(center.x, center.y, 'o', color=color, markersize=8, zorder=11)

# =============================================================================
# set properties for plot
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif', size=10, weight='medium')

# add legend
legend_patches = [
    mpatches.Patch(color=color, label=label) 
            for label, color in branch_color.items()
]

ax.legend(handles=legend_patches, loc='upper left', facecolor='White', 
        fontsize=15, framealpha=0.8, handlelength=1, handletextpad=0.75, ncol=1, 
        borderpad=0.75, columnspacing=1, edgecolor="black", frameon=True,
        bbox_to_anchor=(0.1, 0.95))


ax.set_facecolor('white')

# save plot
fig_name = '/network.pdf'
fig.savefig(_fig_dir + fig_name, dpi=1000)

plt.tight_layout()
plt.show()