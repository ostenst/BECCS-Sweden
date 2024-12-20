"""Stuff here controller"""

import numpy as np
from chp_model import *
import pandas as pd
import seaborn as sns
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
plants_df = pd.read_csv("CHP data all.csv",delimiter=";")
# plants_df = plants_df.iloc[0].to_frame().T # This row makes us only iterate over the 1st plant
# plants_df = plants_df.iloc[:3] # This row makes us only iterate over the 4 first plant

all_experiments = pd.DataFrame()
all_outcomes = pd.DataFrame()

# Load CHP Aspen data
aspen_df = pd.read_csv("MEA-chip.csv", sep=";", decimal=',')
aspen_interpolators = create_interpolators(aspen_df)

for index, plant_data in plants_df.iterrows():

    print(f"||| MODELLING {plant_data['Plant Name']} BIOMASS CHP |||")

    energybalance_assumptions = {
        # "U": 1500                        #[W/m2K]
        # "m_fluegas": simplified from Tharun's study
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
        RealParameter("dTreb", 7, 14),       
        RealParameter("Tsupp", 78, 100),
        RealParameter("Tlow", 43, 55),
        RealParameter("dTmin", 5, 12),       
        RealParameter("U", 1300, 1700),
        RealParameter("COP", 2.3, 3.8),

        RealParameter("alpha", 6, 7),
        RealParameter("beta", 0.6, 0.7),
        RealParameter("CEPCI", 1.386, 1.57), #NOTE: Indices of 780 to 830 in 2026 yield 1.386 and 1.57 (comparing with index of 541 in the reference year 2016 (choosing ref year based on CAPEX function)) 
        RealParameter("fixed", 0.04, 0.08),
        RealParameter("ownercost", 0.1, 0.3),
        RealParameter("WACC", 0.03, 0.09),
        IntegerParameter("yexpenses", 3, 6),
        RealParameter("rescalation", 0.00, 0.06),
        RealParameter("i", 0.05, 0.12),
        IntegerParameter("t", 20, 30),
        RealParameter("celc", 20, 160),
        RealParameter("cheat", 0.25, 1.00), #NOTE: a percentage of celc
        RealParameter("cbio", 15, 60),
        RealParameter("cMEA", 1.5, 2.5),
        RealParameter("cHP", 0.76, 0.96),
        RealParameter("cHEX", 0.470, 0.670),  

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
        if all(df_experiments.index == df_outcomes.index):
            print(" ")
    else:
        print("Mismatch in the number of rows between df_experiments and df_outcomes.")

#     df_outcomes["duration_increase"] = experiments["duration_increase"]
#     # sns.pairplot(df_outcomes, hue="SupplyStrategy", vars=list(outcomes.keys())) # This plots ALL outcomes
#     sns.pairplot(df_outcomes, hue="duration_increase", vars=["capture_cost","penalty_services","penalty_biomass"])

# plt.show()