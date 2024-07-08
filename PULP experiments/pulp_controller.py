"""Stuff here controller"""
# import math
# import numpy as np
# from scipy.optimize import brentq

# import matplotlib.pyplot as plt  
# from scipy.interpolate import LinearNDInterpolator
# from ema_workbench.em_framework.evaluators import Samplers
import seaborn as sns
import pandas as pd
from functions import *
from ema_workbench import (
    Model,
    RealParameter,
    IntegerParameter,
    CategoricalParameter,
    ScalarOutcome,
    Constant,
    Samplers,
    ema_logging,
    perform_experiments
)
from pulp_model import (CCS_Pulp,PulpPlant)

# ------------------ Read data and initiate a plant ----------------------------------
plants_df = pd.read_csv("Pulp data.csv",delimiter=";")
plant_data = plants_df.iloc[0]
print(f"||| MODELLING {plant_data['Name']} PULP MILL |||")
interpolations = ["Interp1", "Interp2"]

energybalance_assumptions = {
    "recovery_intensity": 18,       #[GJ/t]
    "bark_intensity": 4.2,          #[GJ/t] (not needed)
    "heat_intensity": 11,           #[GJ/t]
    "electricity_intensity": 0.7,   #[MWh/t]
    "condensing_pressure": 0.1,     #[bar]
    "time": 8000                    #[h/yr]
}

pulp_plant = PulpPlant(
    name=plant_data['Name'],
    pulp_capacity=plant_data['Pulp capacity'],
    bark_share=plant_data['Bark capacity'],
    recovery_boiler=plant_data['Recovery boiler'],
    bark_boiler=plant_data['Bark boiler'],
    heat_demand=plant_data['Heat demand'],
    electricity_demand=plant_data['Electricity demand'],
    rp=plant_data['RP'],
    rt=plant_data['RT'],
    bp=plant_data['BP'],
    bt=plant_data['BT'],
    lp=plant_data['LP'],
    energybalance_assumptions=energybalance_assumptions
)
pulp_plant.estimate_nominal_cycle() 

# ------------------ Begin RDM analysis  ---------------------------------------------
model = Model("CCSproblem", function=CCS_Pulp)
model.uncertainties = [
    RealParameter("factor_recovery", 0.38, 0.44),       #[tCO2/MWh]
    RealParameter("factor_bark", 0.30, 0.34),
    RealParameter("fluegas_intensity", 10000, 11000),   #[kg/t]
    RealParameter("COP", 2.3, 3.8),
    RealParameter("k", -32, -28),
    RealParameter("m", 1.0, 1.5),

    RealParameter("alpha", 6, 7),
    RealParameter("beta", 0.6, 0.7),
    RealParameter("CEPCI", 1.0, 1.2),
    RealParameter("fixed", 0.04, 0.08),
    RealParameter("ownercost", 0.1, 0.4),
    RealParameter("WACC", 0.03, 0.09),
    IntegerParameter("yexpenses", 2, 6),
    RealParameter("rescalation", 0.02, 0.06),
    RealParameter("i", 0.05, 0.011),
    IntegerParameter("t", 20, 30),
    RealParameter("celc", 20, 100),
    RealParameter("cbio", 30, 150),
    RealParameter("cMEA", 1.5, 2.5),
    RealParameter("cHP", 0.76, 0.96), 
]
model.levers = [
    CategoricalParameter("SupplyStrategy", ["SteamHP", "SteamLP","HeatPumps"]),
    RealParameter("rate", 0.78, 0.94),
    CategoricalParameter("BarkIncrease", ["0","30","60","90"]),
]
model.outcomes = [
    ScalarOutcome("capture_cost", ScalarOutcome.MINIMIZE),
    ScalarOutcome("penalty_services", ScalarOutcome.MINIMIZE),
    ScalarOutcome("penalty_biomass", ScalarOutcome.MINIMIZE),
]
model.constants = [
    Constant("pulp_interpolation", interpolations),
    Constant("PulpPlant", pulp_plant),
]

ema_logging.log_to_stderr(ema_logging.INFO)
n_scenarios = 30
n_policies = 3

results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
experiments, outcomes = results

# ------------------ Process results  ---------------------------------------------
df_experiments = pd.DataFrame(experiments)
df_experiments["Name"] = pulp_plant.name
df_experiments.to_csv("experiments.csv", index=False)

df_outcomes = pd.DataFrame(outcomes)
df_outcomes["Name"] = pulp_plant.name
df_outcomes.to_csv("outcomes.csv", index=False)

df_outcomes["policy"] = experiments["policy"]
sns.pairplot(df_outcomes, hue="policy", vars=list(outcomes.keys()))
plt.show()