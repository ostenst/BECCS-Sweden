import pandas as pd
import matplotlib.pyplot as plt

# Read data
chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

w2e_experiments = pd.read_csv("WASTE experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
w2e_outcomes = pd.read_csv("WASTE experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

pulp_experiments = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

# Combine the outcomes dataframes
combined_outcomes = pd.concat([chp_outcomes, w2e_outcomes, pulp_outcomes])

print(combined_outcomes)

grouped = combined_outcomes.groupby("Name")["capture_cost"].mean()

sorted_grouped = grouped.sort_values()

print(sorted_grouped)