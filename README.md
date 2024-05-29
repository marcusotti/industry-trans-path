# IndustryTransPath
## Python Scripts:
### model.py
model with included model runs with power-mix 0%-100% decarbonized and every availability (pos) and obligation (obl) combination (44 runs)

### plot_network.py
Plot the industrial locations and NUTS regions

### plot_ccs.py
Plot the figure for total carbon emissions captured per case

### plot_results.py
iterate for every case.
plot energy carrier demand for every sector
plot total energy carrier demand
plot import ratio
plot carbon emission reduction and pillars

## Data
### params.xlsx
input parameter

#### SITE 
* output [t]
* SCe [$\tfrac{tCO_2}{t_{out}}$]
* SEC [$\tfrac{MWh}{t_{out}}$]
* elec/NG/coal/H2/alt [1]

#### TECHNOLOGY
* costs [$\tfrac{€}{t_{out}}$]

#### $CO_2$
* price [$\tfrac{€}{t_{out}}$]

#### energy price
* [$\tfrac{€}{MWh}$]

#### max
* [MWh]

## Results
find the power mix (mix0), possibilitiy or obligation of biomethane (ggpos or ggobl) and hydrogen (h2pos or h2obl) in the name of the folder.
Results for annual hydrogen production 0.5 TWh, 2TWh and 3.5TWh saved separately. 2TWh used as default results.

### b_tech
ratio of transition technology active in a year [1]

### costs
[Bn. €]

* Investment costs
* Energy carrier costs
* Carbon emission costs

### v_abated
emissions abated in a year because of a technology [$t_{CO_2}$]

### v_demand
annual energy carrier demand in a site [MWh]

### v_emissions
annual carbon emissions in a site [$t_{CO_2}$]

### v_inv_tech
annual investment costs in a site [€]

### v_mix
annual energy carrier demand domestically produced (green) or imported (grey) [MWh]

* elec, alt always domestic
* NG, coal always imported
* H2, GG cost driven


### v_notcovered
uncovered energy carrier demand in a site [MWh]

### v_sce
specific carbon emissions in a site in a year [$\tfrac{tCO_2}{t_{out}}$]

### v_sec
specific energy consumption in a site in a year [$\tfrac{MWh}{t_{out}}$]