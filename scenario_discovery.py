from ema_workbench.analysis import prim
import pandas as pd
import matplotlib.pyplot as plt

# Read data
# chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
# chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

chp_experiments = pd.read_csv("WASTE experiments/all_experiments.csv",delimiter=",", encoding='utf-8')  #NOTE: SELECT WASTE OR CHP?
chp_outcomes = pd.read_csv("WASTE experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

pulp_experiments = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

# If you want a specific plant
pulp_experiments = pulp_experiments[ (pulp_experiments["Name"] == "Ostrand") ].reset_index(drop=True)
pulp_outcomes = pulp_outcomes[ (pulp_outcomes["Name"] == "Ostrand") ].reset_index(drop=True)
# chp_experiments = chp_experiments[ (chp_experiments["Name"] == "Igelsta KVV") ].reset_index(drop=True)
# chp_outcomes = chp_outcomes[ (chp_outcomes["Name"] == "Igelsta KVV") ].reset_index(drop=True)

# If you want only zero biomass scenarios
zero_biomass_boolean = (pulp_experiments["BarkIncrease"] == 0)
if True:
    pulp_experiments = pulp_experiments[zero_biomass_boolean].reset_index(drop=True)
    pulp_outcomes = pulp_outcomes[zero_biomass_boolean].reset_index(drop=True)

filtered_experiments = pulp_experiments
filtered_outcomes = pulp_outcomes

# #Figure out what plants have above 300 kt/yr
# grouped = chp_outcomes.groupby('Name')
# gross_means = grouped['gross'].mean().sort_values()
# high_gross_names = gross_means[(gross_means > 350)].index.tolist()
# # high_gross_names = gross_means[(gross_means < 150)].index.tolist()
# # high_gross_names = gross_means[(gross_means < 350) & (gross_means > 150)].index.tolist()
# boolean = chp_outcomes['Name'].isin(high_gross_names)
# filtered_outcomes = chp_outcomes[boolean].reset_index(drop=True)
# filtered_experiments = chp_experiments[boolean].reset_index(drop=True)

# Define X and Y
x = filtered_experiments.iloc[:, 0:23]
names_list = ['celc', 'beta', 'SupplyStrategy']  # This is essentially "constrained PRIM"
x = x[names_list]
y = (filtered_outcomes["capture_cost"] < 80) & (filtered_outcomes["penalty_services"] < 450) & (filtered_outcomes["penalty_biomass"] < 200)
print(y.sum(),"scenarios are satisficing out of", len(y))

prim_alg = prim.Prim(x, y, threshold=0.8, peel_alpha=0.1) # Threshold was 0.8 before (Kwakkel) #NOTE: To avoid deprecated error, I replaced line 1506 in prim.py with: np.int(paste_value) => int(paste_value)
box1 = prim_alg.find_box()

box1.show_tradeoff()        # Pareto tradeoff 
box1.write_ppt_to_stdout()  # Prints trajectory/tradeoff, useful!

box1.select(19)
# box1.inspect_tradeoff()     # Print tradeoff to terminal, not useful?
prim_alg.show_boxes(19) 
box1.inspect(19)

# plt.show()