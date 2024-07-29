"""Stuff here controller"""

# import math
import numpy as np
# from scipy.optimize import brentq
# from scipy.interpolate import LinearNDInterpolator
# from ema_workbench.em_framework.evaluators import Samplers
from chp_model import *
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
# from prim_constrained import *

# -------------------------------------- Read data and initiate a plant ----------------------------------
plants_df = pd.read_csv("CHP data all.csv",delimiter=";")
# plants_df = plants_df.iloc[0].to_frame().T # This row makes us only iterate over the 1st plant
all_experiments = pd.DataFrame()
all_outcomes = pd.DataFrame()

# Load CHP Aspen data
aspen_df = pd.read_csv("MEA-chip.csv", sep=";", decimal=',')
aspen_interpolators = create_interpolators(aspen_df)

for index, plant_data in plants_df.iterrows():

    print(f"||| MODELLING {plant_data['Plant Name']} BIOMASS CHP |||")

    energybalance_assumptions = {
        # "time": 5500,                    #[h/yr]
        "U": 1500                        #[W/m2K]
        # "m_fluegas": simplified from Tharun's study
        # "HEX costs": taken from Eliasson (2022)
    }

    CHP = CHP_plant(
        name=plant_data["Plant Name"],
        fuel=plant_data["Fuel (W=waste, B=biomass)"],
        Qdh=plant_data["Heat output (MWheat)"],
        P=plant_data["Electric output (MWe)"],
        Qfgc=plant_data["Existing FGC heat output (MWheat)"],
        ybirth=plant_data["Year of commissioning"],
        Tsteam=plant_data["Live steam temperature (degC)"],
        psteam=plant_data["Live steam pressure (bar)"],
        energybalance_assumptions=energybalance_assumptions
    )
    CHP.estimate_nominal_cycle() 

    # ----------------------------------------- Begin RDM analysis  ---------------------------------------------
    model = Model("CCSproblem", function=CCS_CHP)
    model.uncertainties = [
        RealParameter("dTreb", 7, 14),       #[tCO2/MWh]
        RealParameter("Tsupp", 78, 100),
        RealParameter("Tlow", 43, 55),       #NOTE: Consider 35C as low (Ramboll, Malmö CCS study)
        RealParameter("COP", 2.3, 3.8),
        RealParameter("dTmin", 5, 12),

        RealParameter("alpha", 6, 7),
        RealParameter("beta", 0.6, 0.7),
        RealParameter("CEPCI", 1.0, 1.2),
        RealParameter("fixed", 0.04, 0.08),
        RealParameter("ownercost", 0.1, 0.4),
        RealParameter("WACC", 0.03, 0.09), # REF https://iopscience.iop.org/article/10.1088/1748-9326/aa67a5/meta Dowell, Inefficient BECCS
        IntegerParameter("yexpenses", 2, 6),
        RealParameter("rescalation", 0.02, 0.06),
        RealParameter("i", 0.05, 0.11),
        IntegerParameter("t", 20, 30),
        RealParameter("celc", 20, 100),
        RealParameter("cheat", 30, 150),        
        RealParameter("cbio", 30, 60),
        RealParameter("cMEA", 1.5, 2.5),
        RealParameter("cHP", 0.76, 0.96),
        RealParameter("cHEX", 0.500, 0.600),  

        RealParameter("time", 4000, 6000),
    ]
    model.levers = [
        CategoricalParameter("duration_increase", ["0", "1000","2000"]),
        RealParameter("rate", 0.78, 0.94),
        CategoricalParameter("heat_pump", [True, False]),
    ]
    model.outcomes = [
        ScalarOutcome("capture_cost", ScalarOutcome.MINIMIZE),
        ScalarOutcome("penalty_services", ScalarOutcome.MINIMIZE),
        ScalarOutcome("penalty_biomass", ScalarOutcome.MINIMIZE),
        ArrayOutcome("costs"),
        ArrayOutcome("emissions"),
    ]
    model.constants = [
        Constant("chp_interpolators", aspen_interpolators),
        Constant("CHP", CHP),
    ]

    ema_logging.log_to_stderr(ema_logging.INFO)
    n_scenarios = 250
    n_policies = 40

    results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
    experiments, outcomes = results

    # ---------------------------------------- Process results  ---------------------------------------------
    df_experiments = pd.DataFrame(experiments)
    df_experiments["Name"] = CHP.name
    
    all_experiments = pd.concat([all_experiments, df_experiments], ignore_index=True)
    all_experiments.to_csv("all_experiments.csv", index=False) 

    processed_outcomes = {} # Multi-dimensional outcomes need to be put into neat columns
    for k, v in outcomes.items():
        if isinstance(v, np.ndarray) and v.ndim > 1: 
            for i in range(v.shape[1]):
                processed_outcomes[v[0,i,0]] = v[:,i,1]
        else:
            processed_outcomes[k] = v

    df_outcomes = pd.DataFrame(processed_outcomes)
    df_outcomes["Name"] = CHP.name

    all_outcomes = pd.concat([all_outcomes, df_outcomes], ignore_index=True)
    all_outcomes.to_csv("all_outcomes.csv", index=False)

    if df_experiments.shape[0] == df_outcomes.shape[0]:
        # print("The number of rows in df_experiments and df_outcomes match.")
        if all(df_experiments.index == df_outcomes.index):
            # print("The indices of df_experiments and df_outcomes are aligned.")
            print(" ")
    else:
        print("Mismatch in the number of rows between df_experiments and df_outcomes.")

    df_outcomes["duration_increase"] = experiments["duration_increase"]
    # sns.pairplot(df_outcomes, hue="SupplyStrategy", vars=list(outcomes.keys())) # This plots ALL outcomes
#     sns.pairplot(df_outcomes, hue="duration_increase", vars=["capture_cost","penalty_services","penalty_biomass"])

# plt.show()