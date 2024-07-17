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
plants_df = pd.read_csv("CHP data.csv",delimiter=";")
plants_df = plants_df.iloc[0].to_frame().T # This row makes us only iterate over the 1st plant
all_experiments = pd.DataFrame()
all_outcomes = pd.DataFrame()

# Load CHP Aspen data
aspen_df = pd.read_csv("MEA-chip.csv", sep=";", decimal=',')
aspen_interpolators = create_interpolators(aspen_df)

for index, plant_data in plants_df.iterrows():

    print(f"||| MODELLING {plant_data['Plant Name']} BIOMASS CHP |||")

    energybalance_assumptions = {
        "time": 5500,                    #[h/yr]
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
        RealParameter("Tlow", 43, 55),       #[kg/t]
        RealParameter("COP", 2.3, 3.8),
        RealParameter("dTmin", 5, 12),

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
    n_scenarios = 150
    n_policies = 15

    results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
    experiments, outcomes = results

    # ---------------------------------------- Process results  ---------------------------------------------
    df_experiments = pd.DataFrame(experiments)
    df_experiments["Name"] = CHP.name
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
    df_outcomes["Name"] = CHP.name
    # df_outcomes.to_csv("outcomes.csv", index=False)
    all_outcomes = pd.concat([all_outcomes, df_outcomes], ignore_index=True)

    # Sanity check to ensure the indices are aligned
    if df_experiments.shape[0] == df_outcomes.shape[0]:
        print("The number of rows in df_experiments and df_outcomes match.")
        if all(df_experiments.index == df_outcomes.index):
            print("The indices of df_experiments and df_outcomes are aligned.")
    else:
        print("Mismatch in the number of rows between df_experiments and df_outcomes.")

    df_outcomes["duration_increase"] = experiments["duration_increase"]
    # sns.pairplot(df_outcomes, hue="SupplyStrategy", vars=list(outcomes.keys())) # This plots ALL outcomes
    sns.pairplot(df_outcomes, hue="duration_increase", vars=["capture_cost","penalty_services","penalty_biomass"])

# Launching PRIM algorithm
# zero_biomass_boolean = (df_experiments["BarkIncrease"] == "0")
y = df_outcomes["capture_cost"] < 10
# y = df_outcomes["penalty_services"]<300 
# y = df_outcomes["penalty_biomass"]==0 
# y = (df_outcomes["capture_cost"] < 90) & (df_outcomes["penalty_services"] < 600) & (df_outcomes["penalty_biomass"] < 220)
# y = (all_outcomes["capture_cost"] < 100) & (all_outcomes["penalty_services"] < 500) 

x = all_experiments.iloc[:, 0:23]
print("The number of interesting cases are:\n", y.value_counts())

# Data = PrimedData(x,y)
# peeling_trajectory = []
# box = Box(id=0)
# box.calculate(Data)

# print(" ISSUE: MY PRIM ALGORITHM DOES NOT RECOGNIZE CATEGORICAL FEATURES; THAT'S WHY IT'S WORSE. MAYBE ADJUST IT AND HARD-CODE WHAT IS CATEGORICAL OR NOT, AS AN ARGUMENT")
# peeling_trajectory = prim_recursive(Data,box,peeling_trajectory,max_iterations=40, constrained_to=None, objective_function="LENIENT2")
# # peeling_trajectory = prim_recursive(Data,box,peeling_trajectory,max_iterations=40, constrained_to=["Cellulosic cost", "Biomass backstop price", "Pricing"], objective_function="LENIENT2")
# peeling_trajectory[7].print_info()
# peeling_trajectory[19].print_info()

# x_values = [box.coverage for box in peeling_trajectory]
# y_values = [box.density for box in peeling_trajectory]
# colors = [box.n_lims for box in peeling_trajectory]
# colors = np.array(colors, dtype=int)
# num_colors = len(set(colors))
# cmap = plt.cm.get_cmap('tab10', num_colors)
# plt.scatter(x_values, y_values, c=colors, cmap=cmap, alpha=0.8)
# plt.xlabel('Coverage')
# plt.ylabel('Density')
# plt.colorbar(label='Number of Limits')


prim_alg = prim.Prim(x, y, threshold=0.6, peel_alpha=0.1) # Threshold was 0.8 before (Kwakkel) #NOTE: To avoid deprecated error, I replaced line 1506 in prim.py with: np.int(paste_value) => int(paste_value)
box1 = prim_alg.find_box()

# plt.clf()
box1.show_ppt()             # Lines tradeoff
box1.show_tradeoff()        # Pareto tradeoff 
box1.write_ppt_to_stdout()  # Prints trajectory/tradeoff, useful!

box1.select(14)
# box1.inspect_tradeoff()     # Print tradeoff to terminal, not useful?
prim_alg.show_boxes(14) 
prim_alg.boxes_to_dataframe() # Save boxes here?
box1.inspect(14)

all_experiments.to_csv("all_experiments.csv", index=False)
all_outcomes.to_csv("all_outcomes.csv", index=False)
plt.show()