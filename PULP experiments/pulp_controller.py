"""Stuff here controller"""

# import math
# import numpy as np
# from scipy.optimize import brentq
# from scipy.interpolate import LinearNDInterpolator
# from ema_workbench.em_framework.evaluators import Samplers
from pulp_model import *
import matplotlib.pyplot as plt  
import seaborn as sns
import pandas as pd
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
from ema_workbench.analysis import prim

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

    # ----------------------------------------- Begin RDM analysis  ---------------------------------------------
    model = Model("CCSproblem", function=CCS_Pulp)
    model.uncertainties = [
        RealParameter("factor_recovery", 0.39, 0.42),       #[tCO2/MWh]
        RealParameter("factor_bark", 0.30, 0.33),
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
        RealParameter("i", 0.05, 0.11),
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
        ArrayOutcome("costs"),
        ArrayOutcome("emissions"),
    ]
    model.constants = [
        Constant("pulp_interpolation", interpolations),
        Constant("PulpPlant", pulp_plant),
    ]

    ema_logging.log_to_stderr(ema_logging.INFO)
    n_scenarios = 250
    n_policies = 40

    results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
    experiments, outcomes = results

    # ---------------------------------------- Process results  ---------------------------------------------
    df_experiments = pd.DataFrame(experiments)
    df_experiments["Name"] = pulp_plant.name
    # df_experiments.to_csv("experiments.csv", index=False) 
    all_experiments = pd.concat([all_experiments, df_experiments], ignore_index=True)

    processed_outcomes = {} # Multi-dimensional outcomes need to be put into neat columns
    for k, v in outcomes.items():
        if isinstance(v, np.ndarray) and v.ndim > 1: 
            for i in range(v.shape[1]):
                processed_outcomes[v[0,i,0]] = v[:,i,1]
        else:
            processed_outcomes[k] = v

    # df_outcomes = pd.DataFrame(outcomes)
    df_outcomes = pd.DataFrame(processed_outcomes)
    df_outcomes["Name"] = pulp_plant.name
    # df_outcomes.to_csv("outcomes.csv", index=False)
    all_outcomes = pd.concat([all_outcomes, df_outcomes], ignore_index=True)

    # Sanity check to ensure the indices are aligned
    if df_experiments.shape[0] == df_outcomes.shape[0]:
        print("The number of rows in df_experiments and df_outcomes match.")
        if all(df_experiments.index == df_outcomes.index):
            print("The indices of df_experiments and df_outcomes are aligned.")
    else:
        print("Mismatch in the number of rows between df_experiments and df_outcomes.")

    df_outcomes["SupplyStrategy"] = experiments["SupplyStrategy"]
    # sns.pairplot(df_outcomes, hue="SupplyStrategy", vars=list(outcomes.keys())) # This plots ALL outcomes
    sns.pairplot(df_outcomes, hue="SupplyStrategy", vars=["capture_cost","penalty_services","penalty_biomass"])

# # Launching PRIM algorithm
# # zero_biomass_boolean = (df_experiments["BarkIncrease"] == "0")
# # y = df_outcomes["capture_cost"]<70
# # y = df_outcomes["penalty_services"]<300 
# # y = df_outcomes["penalty_biomass"]==0 
# # y = (df_outcomes["capture_cost"] < 90) & (df_outcomes["penalty_services"] < 600) & (df_outcomes["penalty_biomass"] < 220)
# y = (all_outcomes["capture_cost"] < 100) & (all_outcomes["penalty_services"] < 500) 

# x = all_experiments.iloc[:, 0:23]
# print("The number of interesting cases are:\n", y.value_counts())

# prim_alg = prim.Prim(x, y, threshold=0.8, peel_alpha=0.1) # Threshold was 0.8 before (Kwakkel)
# box1 = prim_alg.find_box()

# plt.clf()
# box1.show_ppt()             # Lines tradeoff
# box1.show_tradeoff()        # Pareto tradeoff 
# box1.write_ppt_to_stdout()  # Prints trajectory/tradeoff, useful!

# box1.select(6)
# box1.inspect_tradeoff()     # Print tradeoff to terminal, not useful?
# prim_alg.show_boxes(6) 
# prim_alg.boxes_to_dataframe() # Save boxes here?
# box1.inspect(6)

all_experiments.to_csv("all_experiments.csv", index=False)
all_outcomes.to_csv("all_outcomes.csv", index=False)
plt.show()