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
from pulp_model import (CCS_Pulp,PulpPlant,State,MEA)

# ------------------ Read data and initiate a plant ----------------------------------
# Load PulpPlant data
# Load PulpAspen data
# Construct a PulpAspenInterpolator here, which will be re-used many times.
interpolations = ["Interp1", "Interp2"]

# Initate a PulpPlant and send it to the model
Plant = PulpPlant("Varo", 900000)
Plant.estimate_nominal_cycle()

# ------------------ Begin RDM analysis  ---------------------------------------------
model = Model("CCSproblem", function=CCS_Pulp)
model.uncertainties = [
    RealParameter("bark_usage", 100, 140),
    IntegerParameter("t", 20, 30),
]
model.levers = [
    CategoricalParameter("EnergySupply", ["Steam","HeatPumps"]),
]
model.outcomes = [
    ScalarOutcome("capture_cost", ScalarOutcome.MINIMIZE),
    ScalarOutcome("penalty_services", ScalarOutcome.MINIMIZE),
    ScalarOutcome("penalty_biomass", ScalarOutcome.MINIMIZE),
]
model.constants = [
            Constant("PulpPlant", Plant),
            Constant("pulp_interpolation", interpolations),
]

ema_logging.log_to_stderr(ema_logging.INFO)
n_scenarios = 30
n_policies = 2

results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
experiments, outcomes = results

# ------------------ Process results  ---------------------------------------------
df_experiments = pd.DataFrame(experiments)
df_experiments["Name"] = Plant.name
df_experiments.to_csv("experiments.csv", index=False)

df_outcomes = pd.DataFrame(outcomes)
df_outcomes["Name"] = Plant.name
df_outcomes.to_csv("outcomes.csv", index=False)

df_outcomes["policy"] = experiments["policy"]
sns.pairplot(df_outcomes, hue="policy", vars=list(outcomes.keys()))
plt.show()