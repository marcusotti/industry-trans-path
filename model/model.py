import pyomo.environ as py
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import math
from utils import print_param, print_var, var2excel

# =============================================================================
# =============================================================================
# =============================================================================
    # ITERATE SCENARIOS
# =============================================================================
# =============================================================================
# =============================================================================
for number in range(4):
    for step in range(11):
        electricity_mix = 0.1 * step

        # =============================================================================
        # =============================================================================
        # =============================================================================
            # CREATE MODEL
        # =============================================================================
        # =============================================================================
        # =============================================================================
        m = py.ConcreteModel()
        m.name = 'INDUSTRY'

        # active scenario
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

        scenario = scenarios[number]
        scenario['name'] = 'mix' + str(step) + '_' + scenario['name']

        _params_dir = os.path.join(os.path.dirname(__file__), 'params.xlsx')
        _results_dir = os.path.join(os.path.dirname(__file__), 'results')
        _results_dir = os.path.join(_results_dir, scenario['name'])
        if not os.path.exists(_results_dir):
            os.makedirs(_results_dir)

        # =============================================================================
        # =============================================================================
        # =============================================================================
            # DEFINE SETS
        # =============================================================================
        # =============================================================================
        # =============================================================================

        # -----------------------------------------------------------------------------
            # SET OF TIMESTEPS: YEARS 2021-2040
        _years = list(range(2021, 2041, 1))
        m.y = py.Set(
            initialize=_years,
            ordered=True,
            doc='set of timesteps in years'
        )

        # -----------------------------------------------------------------------------
            # SET OF TECHNOLOGIES
        technology_df = pd.read_excel(_params_dir, sheet_name='TECHNOLOGY', 
            index_col='technology', header=0)
        technology_df = technology_df.drop(columns=['description'])

        m.tech = py.Set(
            initialize=technology_df.index,
            ordered=True,
            doc='set of technologies'
        )

        # -----------------------------------------------------------------------------
            # SET OF INDUSTRIAL SITES
        site_df = pd.read_excel(_params_dir, sheet_name='SITE', index_col='site', 
            header=0)
        site_df = site_df.drop(columns=['comments'])

        m.site = py.Set(
            initialize=site_df.index,
            ordered=True,
            doc='set of industrial sites'
        )

        # -----------------------------------------------------------------------------
            # SET OF ENERGY CARRIERS
        energy_carrier = ['elec', 'NG', 'coal', 'H2', 'alt', 'GG']

        m.energy_carrier = py.Set(
            initialize=energy_carrier,
            ordered=True,
            doc='set of energy carriers'
        )

        # -----------------------------------------------------------------------------
            # SET OF ENERGY CARRIERS WITHOUT GREEN GAS
        ec_no_gg = {ec for ec in m.energy_carrier if ec != 'GG'}

        m.ec_no_gg = py.Set(
            initialize=ec_no_gg,
            ordered=True,
            doc='set of energy carriers without green gas'
        )

        # -----------------------------------------------------------------------------
            # SET OF ENERGY MIX
        energy_price_2020_df = pd.read_excel(_params_dir,sheet_name='energy price 2020',
            index_col='mix', header=0)

        m.mix = py.Set(
            initialize=energy_price_2020_df.index,
            ordered=True,
            doc='set of energy mix'
        )

        # =============================================================================
        # =============================================================================
        # =============================================================================
            # DEFINE PARAMETERS
        # =============================================================================
        # =============================================================================
        # =============================================================================

        # -----------------------------------------------------------------------------
        # ----- SITE SPECIFIC PARAMETERS ----------------------------------------------
        # BRANCH OF A SITE
        m.p_branch_site = py.Param(
            m.site,
            initialize=site_df['branch'],
            within=py.Any,
            doc='branch of an industrial site'
        )

        # ANNUAL OUTPUT OF AN INDUSTRIAL SITE
        def init_output(m, year, site):
            growth = 0
            return site_df.loc[site, 'output'] * (1 + growth) ** (year - min(m.y))

        m.p_output = py.Param(
            m.y * m.site,
            initialize=init_output,
            within=py.NonNegativeReals,
            doc='annual output of an industrial site in t/a'
        )

        # SPECIFIC ENERGY CONSUMPTION (SEC) OF A SITE
        m.p_sec_site = py.Param(
            m.site,
            initialize=site_df['SEC'],
            within=py.NonNegativeReals,
            doc='specific energy consumption of a site in MWh/t_output'
        )

        # SHARE OF CARRIER OF AN INDUSTRIAL SITE
        m.p_energycarrier_site = py.Param(
            m.site * m.ec_no_gg,
            initialize=site_df[m.ec_no_gg].stack().to_dict(),
            within=py.NonNegativeReals,
            doc='default sec of an industrial site per carrier in MWh/t_output'
        )

        # SCe OF AN INDUSTRIAL SITE
        m.p_sce_site = py.Param(
            m.site,
            initialize=site_df['SCe'],
            within=py.NonNegativeReals,
            doc='default sce of an industrial site in t_emissions/t_output'
        )

        # -----------------------------------------------------------------------------
        # ----- TECHNOLOGY SPECIFIC PARAMETERS ----------------------------------------
        # POSSIBLE BRANCH OF A TECH
        m.p_branch_tech = py.Param(
            m.tech,
            initialize=technology_df['branch'],
            within=py.Any,
            doc='branch of a transition technology'
        )

        # PILLARS OF TRANSITION TECHNOLOGIES
        m.p_pillar = py.Param(
            m.tech,
            initialize=technology_df['pillar'],
            within=py.NonNegativeIntegers,
            doc='pillar of transition technology per tech'
        )

        # TECHNOLOGY READINESS LEVEL
        m.p_trl = py.Param(
            m.tech,
            initialize=technology_df['TRL'],
            within=py.NonNegativeIntegers,
            doc='technology readiness level per tech'
        )

        # NUMBER OF YEARS NEEDED TO INCREASE TRL
        m.p_trlstep = py.Param(
            m.tech,
            initialize=technology_df['TRL step'],
            within=py.NonNegativeReals,
            doc='number of years needed to increase trl per tech'
        )

        # CHANGE OF SPECIFIC ENERGY CONSUMPTION (SEC) WITH A TECH
        def init_sec_tech(m, tech):
            if math.isnan(technology_df.loc[tech]['SEC']):
                return 0
            else:
                return technology_df.loc[tech]['SEC']

        m.p_sec_tech = py.Param(
            m.tech,
            initialize=init_sec_tech,
            within=py.Reals,
            doc='change of specific energy consumption per tech'
        )

        # SHARE OF ENERGY CARRIERS WITH A TECH INSTALLED
        def init_energycarrier_tech(m, tech, energy_carrier):
            if math.isnan(technology_df.loc[tech][energy_carrier]):
                return 0
            else:
                return technology_df.loc[tech][energy_carrier]

        m.p_energycarrier_tech = py.Param(
            m.tech * m.ec_no_gg,
            initialize=init_energycarrier_tech,
            within=py.Reals,
            doc='change of usage of the different energy carriers with a technology'
        )

        # CHANGE OF SPECIFIC CO2 EMISSIONS (SCe) WITH A TECH
        m.p_electricitymix = py.Param(
            initialize = electricity_mix,
            within = py.NonNegativeReals,
            doc = 'Percentage of reaching a decarbonized elec mix by 2030'
        )

        sce_2020_df = pd.read_excel(_params_dir, sheet_name='SCe_2020',
            index_col='technology', header=0)
        sce_2030_df = pd.read_excel(_params_dir, sheet_name='SCe_2030',
            index_col='technology', header=0)

        def init_sce_tech(m, y, tech):
            sce_2020 = sce_2020_df.loc[tech]['SCe']
            sce_2030 = sce_2030_df.loc[tech]['SCe']
            if y <= 2029:
                return (
                    sce_2020 + m.p_electricitymix * (sce_2030 - sce_2020) * 
                    (y - m.y.first()) / 10
                )
            else:
                return sce_2020 + m.p_electricitymix * (sce_2030 - sce_2020)

        m.p_sce_tech = py.Param(
            m.y * m.tech,
            initialize=init_sce_tech,
            within=py.NonNegativeReals,
            doc='change of specific CO2 emissions per year and tech'
        )

        # -----------------------------------------------------------------------------
        # ----- COST PARAMETERS -------------------------------------------------------
        # TECHNOLOGY SPECIFIC INVESTMENT COSTS
        m.c_inv_tech = py.Param(
            m.tech,
            initialize=technology_df['costs'],
            within=py.NonNegativeReals,
            doc='technology specific investment costs per tech in EUR/t_output'
        )

        # ENERGY PRICE
        energy_price_2025_df = pd.read_excel(_params_dir,sheet_name='energy price 2025',
            index_col='mix', header=0)
        energy_price_2030_df = pd.read_excel(_params_dir,sheet_name='energy price 2030',
            index_col='mix', header=0)
        energy_price_2035_df = pd.read_excel(_params_dir,sheet_name='energy price 2035',
            index_col='mix', header=0)

        def init_energy_price(m, y, carrier, mix):
            if y <= 2024:
                return energy_price_2020_df.loc[mix][carrier]
            elif y <= 2029:
                return energy_price_2025_df.loc[mix][carrier]
            elif y <= 2034:
                return energy_price_2030_df.loc[mix][carrier]
            else:
                return energy_price_2035_df.loc[mix][carrier]

        m.c_energy = py.Param(
            m.y * m.energy_carrier * m.mix,
            initialize=init_energy_price,
            within=py.NonNegativeReals,
            doc='energy price per year and energy carrier in EUR/MWh'
        )

        # COSTS OF CO2 EMISSIONS
        co2price_df = pd.read_excel(_params_dir, sheet_name='CO2', index_col='year',
            header=0)

        m.c_co2 = py.Param(
            m.y,
            initialize=co2price_df['price'],
            within=py.NonNegativeReals,
            doc='costs of CO2 emissions per year in EUR/t_CO2'
        )

        # WACC
        m.p_wacc = py.Param(
            initialize=0.065,
            within=py.NonNegativeReals,
            doc='weighted average cost of capital.'
        )

        # PENALTY FOR NOT COVERING THE DEMAND OF A SITE
        m.p_penalty = py.Param(
            initialize=1e18,
            within=py.NonNegativeReals,
            doc='penalty for not covering the the demand of a site'
        )

        # -----------------------------------------------------------------------------
        # ----- LOCATION SPECIFIC PARAMETERS ------------------------------------------
        # MAXIMAL AVAILABILITY OF AN ENERGY CARRIER IN THE SYSTEM
        max_2020_df = pd.read_excel(_params_dir, sheet_name='max 2020', 
            index_col='mix', header=0)
        max_2025_df = pd.read_excel(_params_dir, sheet_name='max 2025', 
            index_col='mix', header=0)
        max_2030_df = pd.read_excel(_params_dir, sheet_name='max 2030',
            index_col='mix', header=0)
        max_2035_df = pd.read_excel(_params_dir, sheet_name='max 2035',
            index_col='mix', header=0)
        max_2040_df = pd.read_excel(_params_dir, sheet_name='max 2040',
            index_col='mix', header=0)

        def init_maxcarrier(m, y, energy_carrier, mix):
            if y <= 2024:
                if math.isnan(max_2020_df.loc[mix][energy_carrier]):
                    return math.inf
                else:
                    return max_2020_df.loc[mix][energy_carrier]
            elif y <= 2029:
                if math.isnan(max_2025_df.loc[mix][energy_carrier]):
                    return math.inf
                else:
                    return max_2025_df.loc[mix][energy_carrier]
            elif y <= 2034:
                if math.isnan(max_2030_df.loc[mix][energy_carrier]):
                    return math.inf
                else:
                    return max_2030_df.loc[mix][energy_carrier]
            elif y <= 2039:
                if math.isnan(max_2035_df.loc[mix][energy_carrier]):
                    return math.inf
                else:
                    return max_2035_df.loc[mix][energy_carrier]
            else:
                if math.isnan(max_2040_df.loc[mix][energy_carrier]):
                    return math.inf
                else:
                    return max_2040_df.loc[mix][energy_carrier]

        m.p_maxcarrier = py.Param(
            m.y * m.energy_carrier * m.mix,
            initialize=init_maxcarrier,
            within=py.NonNegativeReals,
            doc='maximal availability of an energy carrier in the system'
        )

        # =============================================================================
        # =============================================================================
        # =============================================================================
            # DEFINE SUBSETS
        # =============================================================================
        # =============================================================================
        # =============================================================================
        # -----------------------------------------------------------------------------
            # SET OF PILLAR 2 OR 3 TECHNOLOGIES
        pillars23 = {tech for tech in m.tech if m.p_pillar[tech] in [2, 3]}

        m.tech23 = py.Set(
            initialize=pillars23,
            ordered=True,
            doc='set of pillar 2 or 3 technologies'
        )

        # =============================================================================
        # =============================================================================
        # =============================================================================
            # DEFINE VARIABLES
        # =============================================================================
        # =============================================================================
        # =============================================================================

        # ANNUAL ENERGY CONSUMPTION OF AN INDUSTRIAL SITE 
        m.v_demand = py.Var(
            m.y * m.site * m.energy_carrier,
            within=py.NonNegativeReals,
            doc='annual energy consumption of a site in MWh'
        )

        # ANNUAL ENERGY DEMAND PER ENERGY MIX
        m.v_mix = py.Var(
            m.y * m.energy_carrier * m.mix,
            within=py.NonNegativeReals,
            doc='annual energy demand per energy mix in MWh'
        )

        # ANNUAL CO2 EMISSIONS OF AN INDUSTRIAL SITE
        m.v_emissions = py.Var(
            m.y * m.site,
            within=py.Reals,
            doc='annual CO2 emissions of a site in t_CO2'
        )

        # SPECIFIC ENERGY CONSUMPTION OF AN INDUSTRIAL SITE
        m.v_sec = py.Var(
            m.y * m.site * m.ec_no_gg,
            within=py.NonNegativeReals,
            doc='specific energy consumption of an industrial site in MWh/t_output'
        )

        # SPECIFIC CO2 EMISSIONS OF AN INDUSTRIAL SITE
        m.v_sce = py.Var(
            m.y * m.site,
            within=py.NonNegativeReals,
            doc='specific CO2 emissions of an industrial site in t_CO2/t_output'
        )

        # VARIABLE FOR INSTALLED TECH
        m.b_tech = py.Var(
            m.y * m.site * m.tech,
            within=py.UnitInterval,
            doc='variable between 0 and 1 for installed tech per year, site and tech'
        )

        # BINARY VARIABLE FOR INSALLED TECH IN PILLAR 2 OR 3
        m.b_tech_binary = py.Var(
            m.y * m.site * m.tech,
            within=py.Binary,
            doc='binary variable for installed tech in pillar 2 or 3'
        )

        #  VARIABLE FOR INVESTMENT DECISION
        m.b_inv = py.Var(
            m.y * m.site * m.tech,
            within=py.UnitInterval,
            doc='binary variable for investment decision per year, site and tech'
        )

        # TOTAL INVESTMENT COST IN TECH IN A SITE
        m.v_inv_tech = py.Var(
            m.y * m.site,
            within=py.NonNegativeReals,
            doc='total investment costs in tech in an industrial site per year in EUR'    
        )

        # UNCOVERED DEMAND OF A SITE
        m.v_notcovered = py.Var(
            m.y * m.site * m.energy_carrier,
            within=py.NonNegativeReals,
            doc='uncovered demand of a site per year in MWh'
        )

        # ABATED EMISSION OF A SITE WITH A TECH
        m.v_abated = py.Var(
            m.y * m.tech,
            within=py.NonNegativeReals,
            doc='abated emissions of a site with a tech per year in t_CO2'
        )

        # =============================================================================
        # =============================================================================
        # =============================================================================
            # DEFINE CONSTRAINTS
        # =============================================================================
        # =============================================================================
        # =============================================================================

        # -----------------------------------------------------------------------------
        # ----- ENERGY DEMAND AND EMISSIONS -------------------------------------------
        # SPECIFIC ENERGY CONSUMPTION OF AN INDUSTRIAL SITE
        def sec(m, y, site, carrier):
            return (
                m.v_sec[y, site, carrier] == 
                m.p_sec_site[site] * (m.p_energycarrier_site[site, carrier] - sum(
                (m.p_energycarrier_site[site, carrier] * m.p_sec_tech[tech] + 
                m.p_energycarrier_tech[tech, carrier] - m.p_sec_tech[tech] *
                m.p_energycarrier_tech[tech, carrier]) * m.b_tech[y, site, tech] 
                for tech in m.tech))
                )

        m.eq_sec = py.Constraint(
            m.y * m.site * m.ec_no_gg,
            rule=sec,
            doc='specific energy consumption of a site per year and carrier'
        )

        # SPECIFIC CO2 EMISSIONS OF AN INDUSTRIAL SITE
        def sce(m, y, site):
            return (
                m.v_sce[y, site] == m.p_sce_site[site] * (1 - sum(
                m.p_sce_tech[y, tech] * m.b_tech[y, site, tech] for tech in m.tech))
                )

        m.eq_sce = py.Constraint(
            m.y * m.site,
            rule=sce,
            doc='specific CO2 emissions of an industrial site per year'
        )

        # NEWLY INSTALLED TECHNOLOGY
        def new_tech(m, y, site, tech):
            if y == m.y.first():
                return m.b_inv[y, site, tech] == m.b_tech[y, site, tech]
            else:
                return (
                    m.b_inv[y, site, tech] >= 
                    m.b_tech[y, site, tech] - m.b_tech[y-1, site, tech]
                    )

        m.eq_new_tech = py.Constraint(
            m.y * m.site * m.tech,
            rule=new_tech,
            doc='newly installed tech per year and site'
        )

        # ANNUAL INVESTMENTS IN TECHNOLOGY OF AN INDUSTRIAL SITE
        def inv_tech(m, y, site):
            return (
                m.v_inv_tech[y, site] == 
                sum(m.b_inv[y, site, tech] * m.c_inv_tech[tech] for tech in m.tech)
                * m.p_output[y, site]
                )

        m.eq_inv_tech = py.Constraint(
            m.y * m.site,
            rule=inv_tech,
            doc='investment in tech of a site per year in EUR'
        )

        # ANNUAL ENERGY CONSUMPTION OF AN INDUSTRIAL SITE
        def demand(m, y, site, carrier):
            if carrier == 'NG' or carrier == 'GG':
                return (
                    sum(m.v_demand[y, site, ec] for ec in ['NG', 'GG']) ==
                    m.v_sec[y, site, 'NG'] * m.p_output[y, site]
                    - sum(m.v_notcovered[y, site, carrier] for ec in ['NG', 'GG'])
                    )
            else:
                return (
                    m.v_demand[y, site, carrier] ==
                    m.v_sec[y, site, carrier] * m.p_output[y, site]
                    - m.v_notcovered[y, site, carrier]
                    )

        m.eq_demand = py.Constraint(
            m.y * m.site * m.energy_carrier,
            rule=demand,
            doc='annual energy consumption of a site'
        )

        # DEMAND IS THE SUM OF ENERGY MIX
        def demand_mix(m, y, carrier):
            return (
                sum(m.v_demand[y, site, carrier] for site in m.site) ==
                sum(m.v_mix[y, carrier, mix] for mix in m.mix)
            )

        m.eq_demand_mix = py.Constraint(
            m.y * m.energy_carrier,
            rule=demand_mix,
            doc='demand in the system is the sum of energy mix'
        )

        # ANNUAL CO2 EMISSIONS OF AN INDUSTRIAL SITE
        def emissions(m, y, site):
            return m.v_emissions[y, site] == m.v_sce[y, site] * m.p_output[y, site]

        m.eq_emissions = py.Constraint(
            m.y * m.site,
            rule=emissions,
            doc='annual CO2 emissions of a site'
        )

        # -----------------------------------------------------------------------------
        # ----- LIMITATIONS OF TRANS TECH ---------------------------------------------
        # LIMIT THE USE OF A TECHNOLOGY TO TRL 
        def limit_trl(m, y, site , tech):
            if m.p_trl[tech] > 8:
                return py.Constraint.Skip
            else:
                if y > (((8 - m.p_trl[tech]) * m.p_trlstep[tech]) + m.y.first()):
                    return py.Constraint.Skip
                else:
                    return m.b_tech[y, site, tech] == 0

        m.eq_limit_trl = py.Constraint(
            m.y * m.site * m.tech,
            rule=limit_trl,
            doc='limit the use of a tech to TRL>8, TRL increase by one every n years'
        )

        # ONLY USE TRANS TECH IN THE SAME BRANCH AS THE SITE
        def limit_techbranch(m, y, site, tech):
            if m.p_branch_tech[tech] == m.p_branch_site[site]:
                return py.Constraint.Skip
            else:
                return m.b_tech[y, site, tech] == 0

        m.eq_limit_techbranch = py.Constraint(
            m.y * m.site * m.tech,
            rule=limit_techbranch,
            doc='limit the use of a tech to the same branch as the site'
        )

        # MAKE SURE THAT b_tech IS BINARY FOR PILLAR 2 OR 3
        def b_tech_bin_rule(m, y, site, tech):
            return m.b_tech[y, site, tech] == m.b_tech_binary[y, site, tech]

        m.eq_b_tech_bin = py.Constraint(
            m.y * m.site * m.tech23,
            rule=b_tech_bin_rule,
            doc='make sure that b_tech is binary for pillar 2 or 3'
        )

        # USE ONLY PILLAR 2 OR 3, NOT BOTH
        def limit_pillar(m, y, site):
            return sum(m.b_tech[y, site, tech] for tech in m.tech23) <= 1

        m.eq_limit_pillar = py.Constraint(
            m.y * m.site,
            rule=limit_pillar,
            doc='limit the use of pillar 2 or 3 to one per site'
        )

        # ONLY USE PILLAR 1 OR 4 FOR I&S SITES WITH EAF INSTALLED
        eaf_sites = ['Marienhütte Graz', 'Böhler Edelstahl Kapfenberg', 
                        'Breitendorf Edelstahl Mitterdorf']
        def limit_eaf(m, y, site, tech):
            if site in eaf_sites:
                return m.b_tech[y, site, tech] == 0
            else:
                return py.Constraint.Skip

        m.eq_limit_eaf = py.Constraint(
            m.y * m.site * m.tech23,
            rule=limit_eaf,
            doc='only use pillar 1 or 4 in sites with EAF'
        )

        # DRI NEED EAF OVEN
        dri = ['DRI-NG', 'DRI-H2']
        def limit_dri(m, y, site, tech):
            if tech in dri:
                return (
                    m.b_tech[y, site, tech] + m.b_tech[y, site, 'EAF'] >=
                    2 * m.b_tech[y, site, tech]
                    )
            else:
                return py.Constraint.Skip

        m.eq_limit_dri = py.Constraint(
            m.y * m.site * m.tech,
            rule=limit_dri,
            doc='DRI needs EAF'
        )

        # DRI CAN ONLY BE REPLACED BY DRI, NOT REMOVED
        def limit_dri_removal(m, y, site):
            if y == m.y.first():
                return py.Constraint.Skip
            else:
                return (
                    sum(m.b_tech[y, site, t] for t in dri) >=
                    sum(m.b_tech[y-1, site, t] for t in dri)
                )
        
        m.eq_limit_dri_removal = py.Constraint(
            m.y * m.site,
            rule=limit_dri_removal,
            doc='DRI can only be replaced by DRI, not removed'
        )

        # SECURE PHASING OUT OF COAL
        def limit_coal(m, y):
            carrier = 'coal'
            if y <= 2024:
                return py.Constraint.Skip
            elif y <= 2025:
                return (
                    sum(m.v_demand[y, site, carrier] for site in m.site) <=
                    sum(m.v_demand[m.y.first(), site, carrier] * 0.9
                    for site in m.site)
                )
            elif y <= 2026:
                return (
                    sum(m.v_demand[y, site, carrier] for site in m.site) <=
                    sum(m.v_demand[m.y.first(), site, carrier] * 0.75
                    for site in m.site)
                )
            elif y <= 2027:
                return (
                    sum(m.v_demand[y, site, carrier] for site in m.site) <=
                    sum(m.v_demand[m.y.first(), site, carrier] * 0.6
                    for site in m.site)
                )
            elif y <= 2028:
                return (
                    sum(m.v_demand[y, site, carrier] for site in m.site) <=
                    sum(m.v_demand[m.y.first(), site, carrier] * 0.45
                    for site in m.site)
                )
            elif y <= 2029:
                return (
                    sum(m.v_demand[y, site, carrier] for site in m.site) <=
                    sum(m.v_demand[m.y.first(), site, carrier] * 0.3
                    for site in m.site)
                )
            elif y <= 2034:
                return (
                    sum(m.v_demand[y, site, carrier] for site in m.site) <=
                    sum(m.v_demand[m.y.first(), site, carrier] * 0.15
                    for site in m.site)
                )
            else:
                return (
                    sum(m.v_demand[y, site, carrier] for site in m.site) <=
                    sum(m.v_demand[m.y.first(), site, carrier] * 0.1
                    for site in m.site)
                )
            
        m.eq_limit_coal = py.Constraint(
            m.y,
            rule=limit_coal,
            doc='secure phasing out of coal'
        )

        # ENERGY CARRIER INCREASE LIMITATION
        def carrier_incr_limit(m, y, carrier):
            max_incr = max(m.p_sec_site[site] * m.p_energycarrier_site[site, carrier]
                * m.p_output[m.y.first(), site] 
                for site in m.site for carrier in m.ec_no_gg)
            if y == m.y.first():
                return py.Constraint.Skip
            else:
                return (
                    sum(m.v_mix[y, carrier, mix] for mix in m.mix) <= 
                    sum(m.v_mix[y-1, carrier, mix] for mix in m.mix)
                    + max_incr * 1.5
                )
            
        m.eq_carrier_incr_limit = py.Constraint(
            m.y * m.ec_no_gg,
            rule=carrier_incr_limit,
            doc='energy carrier increase limitation'
        )

        # GREEN GASES INCREASE LIMITATION
        def gg_incr_limit(m, y):
            if y == m.y.first():
                return py.Constraint.Skip
            else:
                return (
                    sum(m.v_mix[y, 'GG', mix] for mix in m.mix) <= 
                    sum(m.v_mix[y-1, 'GG', mix] for mix in m.mix)
                    + 5000000
                )
        
        m.eq_gg_incr_limit = py.Constraint(
            m.y,
            rule=gg_incr_limit,
            doc='green gases increase limitation'
        )

        # NO NEW TECH IN FIRST YEAR
        def yearzero(m, site, tech):
            return m.b_tech[m.y.first(), site, tech] == 0

        m.eq_yearzero = py.Constraint(
            m.site * m.tech,
            rule=yearzero,
            doc='no new tech in first year'
        )

        # NO UNCOVERED DEMAND IN THE FIRST YEAR
        def yearzero_demand(m, site, carrier):
            return m.v_notcovered[m.y.first(), site, carrier] == 0
        
        m.eq_yearzero_demand = py.Constraint(
            m.site * m.energy_carrier,
            rule=yearzero_demand,
            doc='no uncovered demand in the first year'
        )

        # -----------------------------------------------------------------------------
        # ----- INFRASTRUCTURAL CONSTRAINTS -------------------------------------------
        # MAXIMAL AVAILABILITY OF AN ENERGY CARRIER IN THE SYSTEM
        def carrier_limit_sys(m, y, carrier, mix):
            return m.v_mix[y, carrier, mix] <= m.p_maxcarrier[y, carrier, mix]

        m.eq_carrier_limit_sys = py.Constraint(
            m.y * m.energy_carrier * m.mix,
            rule=carrier_limit_sys,
            doc='maximal availability of an energy carrier in the system'
        )

        # OBLIGATION TO USE NATIONAL CARRIERS
        def h2_obligation(m, y):
            return (
                m.v_mix[y, 'H2', 'green'] >= 
                m.p_maxcarrier[y, 'H2', 'green']
                )

        m.eq_h2_obligation = py.Constraint(
            m.y,
            rule=h2_obligation,
            doc='obligation to use national carriers'
        )

        if not scenario['h2']:
            m.eq_h2_obligation.deactivate()

        # OBLIGATION TO USE GREEN GAS
        def gg_obligation(m, y):
            return (
                m.v_mix[y, 'GG', 'green'] >= 
                m.p_maxcarrier[y, 'GG', 'green']
                )

        m.eq_gg_obligation = py.Constraint(
            m.y,
            rule=gg_obligation,
            doc='obligation to use green gas'
        )

        if not scenario['gg']:
            m.eq_gg_obligation.deactivate()

        # CALCULATE ABATED EMISSIONS
        def abated_emissions(m, y, tech):
            return(
                m.v_abated[y, tech] == 
                sum(m.b_tech[y, site, tech] * m.p_sce_site[site] * m.p_output[y, site]
                * m.p_sce_tech[y, tech] for site in m.site)
            )

        m.eq_abated_emissions = py.Constraint(
            m.y * m.tech,
            rule=abated_emissions,
            doc='abated emissions of a site with a tech'
        )

        # =============================================================================
        # =============================================================================
        # =============================================================================
            # DEFINE OBJECTIVE FUNCTION
        # =============================================================================
        # =============================================================================
        # =============================================================================

        def objective_function(m):
            npv = 0

            for y in m.y:
                value = 0
                year = y - m.y.first()  

                value += sum(
                    m.v_inv_tech[y, site]
                    + sum(
                        m.v_notcovered[y, site, carrier] * m.p_penalty
                        for carrier in m.energy_carrier
                        )
                    + m.v_emissions[y, site] * m.c_co2[y]
                    for site in m.site
                )

                value += sum(
                    m.v_mix[y, carrier, mix] * m.c_energy[y, carrier, mix]
                    for carrier in m.energy_carrier for mix in m.mix
                )
            
                npv += value / (1 + m.p_wacc) ** year
            return npv

        m.obj_func = py.Objective(expr=objective_function, sense=py.minimize)

        # =============================================================================
        # =============================================================================
        # =============================================================================
            # SOLVE MODEL
        # =============================================================================
        # =============================================================================
        # =============================================================================

        Solver = py.SolverFactory('gurobi')
        Solver.options['LogFile'] = os.path.join(
            os.path.dirname(__file__), str(m.name) + '.log')
        Solver.options['mipgap'] = 0.01
        Solver.options['MIPFocus'] = 1
        solution = Solver.solve(m, tee=True)

        # =============================================================================
        # =============================================================================
        # =============================================================================
            # SAVE RESULTS
        # =============================================================================
        # =============================================================================
        # =============================================================================
        # write results to excel files
        var2excel(m.v_demand, scenario, m.y, m.site, m.energy_carrier)
        var2excel(m.v_sec, scenario, m.y, m.site, m.ec_no_gg)
        var2excel(m.v_sce, scenario, m.y, m.site)
        var2excel(m.v_mix, scenario,  m.y, m.energy_carrier, m.mix)
        var2excel(m.b_tech, scenario,  m.y, m.site, m.tech)
        var2excel(m.v_notcovered, scenario,  m.y, m.site, m.energy_carrier)
        var2excel(m.v_inv_tech, scenario,  m.y, m.site)
        var2excel(m.v_emissions, scenario,  m.y, m.site)
        var2excel(m.v_abated, scenario,  m.y, m.tech)

        # print total costs
        CAPEX = sum(m.v_inv_tech[y, site].value for y in m.y for site in m.site)
        OPEX_dem = sum(
            m.v_mix[y, carrier, mix].value * m.c_energy[y, carrier, mix] 
            for y in m.y for mix in m.mix for carrier in m.energy_carrier
            )
        OPEX_emis = sum(
            m.v_emissions[y, site].value * m.c_co2[y]
            for y in m.y for site in m.site
            )

        with open(os.path.join(_results_dir, 'costs.txt'), 'w') as file:
            file.write('Total CAPEX: ' + str(CAPEX/1e9) + '\n')
            file.write('Total OPEX_dem: ' + str(OPEX_dem/1e9) + '\n')
            file.write('Total OPEX_emis: ' + str(OPEX_emis/1e9) + '\n')

        # print costs for IS
        CAPEX = sum(m.v_inv_tech[y, site].value for y in m.y for site in m.site if m.p_branch_site[site] == 'IS')
        OPEX_dem = 0
        for y in m.y:
            for c in m.energy_carrier:
                if sum(m.v_mix[y, c, mix].value for mix in m.mix) > 0:
                    mix_green = m.v_mix[y, c, 'green'].value / sum(m.v_mix[y, c, mix].value for mix in m.mix)
                else:
                    mix_green = 0
                OPEX_dem += sum(m.v_demand[y,s,c].value for s in m.site if m.p_branch_site[s] == 'IS') * m.c_energy[y, c, 'green'] * mix_green
                OPEX_dem += sum(m.v_demand[y,s,c].value for s in m.site if m.p_branch_site[s] == 'IS') * m.c_energy[y, c, 'grey'] * (1 - mix_green)
        OPEX_emis = sum(
            m.v_emissions[y, site].value * m.c_co2[y]
            for y in m.y for site in m.site if m.p_branch_site[site] == 'IS'
            )

        with open(os.path.join(_results_dir, 'costs.txt'), 'a') as file:
            file.write('IS CAPEX: ' + str(CAPEX/1e9) + '\n')
            file.write('IS OPEX_dem: ' + str(OPEX_dem/1e9) + '\n')
            file.write('IS OPEX_emis: ' + str(OPEX_emis/1e9) + '\n')

        # print costs for PP
        CAPEX = sum(m.v_inv_tech[y, site].value for y in m.y for site in m.site if m.p_branch_site[site] == 'PP')
        OPEX_dem = 0
        for y in m.y:
            for c in m.energy_carrier:
                if sum(m.v_mix[y, c, mix].value for mix in m.mix) > 0:
                    mix_green = m.v_mix[y, c, 'green'].value / sum(m.v_mix[y, c, mix].value for mix in m.mix)
                else:
                    mix_green = 0
                OPEX_dem += sum(m.v_demand[y,s,c].value for s in m.site if m.p_branch_site[s] == 'PP') * m.c_energy[y, c, 'green'] * mix_green
                OPEX_dem += sum(m.v_demand[y,s,c].value for s in m.site if m.p_branch_site[s] == 'PP') * m.c_energy[y, c, 'grey'] * (1 - mix_green)
        OPEX_emis = sum(
            m.v_emissions[y, site].value * m.c_co2[y]
            for y in m.y for site in m.site if m.p_branch_site[site] == 'PP'
            )

        with open(os.path.join(_results_dir, 'costs.txt'), 'a') as file:
            file.write('PP CAPEX: ' + str(CAPEX/1e9) + '\n')
            file.write('PP OPEX_dem: ' + str(OPEX_dem/1e9) + '\n')
            file.write('PP OPEX_emis: ' + str(OPEX_emis/1e9) + '\n')

        # print costs for NMM
        CAPEX = sum(m.v_inv_tech[y, site].value for y in m.y for site in m.site if m.p_branch_site[site] == 'NMM')
        OPEX_dem = 0
        for y in m.y:
            for c in m.energy_carrier:
                if sum(m.v_mix[y, c, mix].value for mix in m.mix) > 0:
                    mix_green = m.v_mix[y, c, 'green'].value / sum(m.v_mix[y, c, mix].value for mix in m.mix)
                else:
                    mix_green = 0
                OPEX_dem += sum(m.v_demand[y,s,c].value for s in m.site if m.p_branch_site[s] == 'NMM') * m.c_energy[y, c, 'green'] * mix_green
                OPEX_dem += sum(m.v_demand[y,s,c].value for s in m.site if m.p_branch_site[s] == 'NMM') * m.c_energy[y, c, 'grey'] * (1 - mix_green)
        OPEX_emis = sum(
            m.v_emissions[y, site].value * m.c_co2[y]
            for y in m.y for site in m.site if m.p_branch_site[site] == 'NMM'
            )

        with open(os.path.join(_results_dir, 'costs.txt'), 'a') as file:
            file.write('NMM CAPEX: ' + str(CAPEX/1e9) + '\n')
            file.write('NMM OPEX_dem: ' + str(OPEX_dem/1e9) + '\n')
            file.write('NMM OPEX_emis: ' + str(OPEX_emis/1e9) + '\n')

        print('##################################################################')
        print('##################################################################')
        print('##################################################################')
        print('Scenario ' + scenario['name'] + ' solved!')
        print('##################################################################')
        print('##################################################################')
        print('##################################################################')