"""Stuff here controller"""

from pulp_model import *
import matplotlib.pyplot as plt  
import seaborn as sns
import pandas as pd
import numpy as np
from ema_workbench import (
    Model,
    RealParameter,
    IntegerParameter,
    CategoricalParameter,
    ScalarOutcome,
    ArrayOutcome,
    Constant,
    Samplers,
    ema_logging,
    perform_experiments
)

# -------------------------------------- Read data and initiate a plant ----------------------------------
plants_df = pd.read_csv("Pulp data.csv",delimiter=";")
all_experiments = pd.DataFrame()
all_outcomes = pd.DataFrame()
# plant_data = plants_df.iloc[5]

for index, plant_data in plants_df.iterrows():
    print(f"||| MODELLING {plant_data['Name']} PULP MILL |||")
    interpolations = ["Interp1", "Interp2"]

    energybalance_assumptions = {
        "recovery_intensity": 18,       #[GJ/t]
        "bark_intensity": 4.2,          #[GJ/t] (not needed)
        "heat_intensity": 11,           #[GJ/t]
        "electricity_intensity": 0.7,   #[MWh/t]
        "condensing_pressure": 0.1,     #[bar]
        "time": 8000                    #[h/yr] NOTE: This cannot be an uncertainty as we are hard-coding the pulp capacities [t/yr], and this capacity determines A LOT!
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

    # ----------------------------------------- Begin RDM analysis  ---------------------------------------------
    model = Model("CCSproblem", function=CCS_Pulp)
    model.uncertainties = [
        RealParameter("factor_recovery", 0.38, 0.43),       #[tCO2/MWh]
        RealParameter("factor_bark", 0.29, 0.34),
        RealParameter("fluegas_intensity", 10000, 11000),   #[kg/t]
        RealParameter("COP", 2.3, 3.8),
        RealParameter("k", -217, 157),
        RealParameter("m", 0.918, 1.578),

        RealParameter("alpha", 6, 7),
        RealParameter("beta", 0.6, 0.7),
        RealParameter("CEPCI", 1.38, 1.57),
        RealParameter("fixed", 0.04, 0.08),
        RealParameter("ownercost", 0.1, 0.3),
        RealParameter("WACC", 0.03, 0.09),
        IntegerParameter("yexpenses", 3, 6),
        RealParameter("rescalation", 0.00, 0.06),
        RealParameter("i", 0.05, 0.12),
        IntegerParameter("t", 20, 30),
        RealParameter("celc", 20, 160),
        RealParameter("cbio", 15, 60),
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
        ArrayOutcome("costs"),
        ArrayOutcome("emissions"),
    ]
    model.constants = [
        Constant("pulp_interpolation", interpolations),
        Constant("PulpPlant", pulp_plant),
    ]

    ema_logging.log_to_stderr(ema_logging.INFO)
    n_scenarios = 500
    n_policies = 50

    results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
    experiments, outcomes = results

    # ---------------------------------------- Process results  ---------------------------------------------
    df_experiments = pd.DataFrame(experiments)
    df_experiments["Name"] = pulp_plant.name
    all_experiments = pd.concat([all_experiments, df_experiments], ignore_index=True)

    processed_outcomes = {} # Multi-dimensional outcomes need to be put into neat columns
    for k, v in outcomes.items():
        if isinstance(v, np.ndarray) and v.ndim > 1: 
            for i in range(v.shape[1]):
                processed_outcomes[v[0,i,0]] = v[:,i,1]
        else:
            processed_outcomes[k] = v

    df_outcomes = pd.DataFrame(processed_outcomes)
    df_outcomes["Name"] = pulp_plant.name
    all_outcomes = pd.concat([all_outcomes, df_outcomes], ignore_index=True)

    # Sanity check to ensure the indices are aligned
    if df_experiments.shape[0] == df_outcomes.shape[0]:
        print("The number of rows in df_experiments and df_outcomes match.")
        if all(df_experiments.index == df_outcomes.index):
            print("The indices of df_experiments and df_outcomes are aligned.")
    else:
        print("Mismatch in the number of rows between df_experiments and df_outcomes.")

    # df_outcomes["SupplyStrategy"] = experiments["SupplyStrategy"]
    # sns.pairplot(df_outcomes, hue="SupplyStrategy", vars=list(outcomes.keys())) # This plots ALL outcomes
    # sns.pairplot(df_outcomes, hue="SupplyStrategy", vars=["capture_cost","penalty_services","penalty_biomass"])

all_experiments.to_csv("all_experiments.csv", index=False)
all_outcomes.to_csv("all_outcomes.csv", index=False)
# plt.show()