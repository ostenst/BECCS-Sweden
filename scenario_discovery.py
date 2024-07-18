from ema_workbench.analysis import prim
import pandas as pd
import matplotlib.pyplot as plt

# Read data
chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

pulp_experiments = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

# If you want a specific plant
pulp_experiments = pulp_experiments[ (pulp_experiments["Name"] == "Morrum") ].reset_index(drop=True)
pulp_outcomes = pulp_outcomes[ (pulp_outcomes["Name"] == "Morrum") ].reset_index(drop=True)

# # If you want only zero biomass scenarios
# zero_biomass_boolean = (pulp_experiments["BarkIncrease"] == 0)
# if True:
#     pulp_experiments = pulp_experiments[zero_biomass_boolean].reset_index(drop=True)
#     pulp_outcomes = pulp_outcomes[zero_biomass_boolean].reset_index(drop=True)

# Define X and Y
x = pulp_experiments.iloc[:, 0:23]
y = (pulp_outcomes["capture_cost"] < 100) & (pulp_outcomes["penalty_services"] < 450) & (pulp_outcomes["penalty_biomass"] < 250)
print(y.sum(),"scenarios are satisficing out of", len(y))

prim_alg = prim.Prim(x, y, threshold=0.6, peel_alpha=0.1) # Threshold was 0.8 before (Kwakkel) #NOTE: To avoid deprecated error, I replaced line 1506 in prim.py with: np.int(paste_value) => int(paste_value)
box1 = prim_alg.find_box()

box1.show_ppt()             # Lines tradeoff
box1.show_tradeoff()        # Pareto tradeoff 
box1.write_ppt_to_stdout()  # Prints trajectory/tradeoff, useful!

box1.select(6)
# box1.inspect_tradeoff()     # Print tradeoff to terminal, not useful?
prim_alg.show_boxes(6) 
box1.inspect(6)

plt.show()