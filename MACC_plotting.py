import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

w2e_experiments = pd.read_csv("WASTE experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
w2e_outcomes = pd.read_csv("WASTE experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

pulp_experiments = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

# Combine the outcomes dataframes, sort by capture_cost [EUR/t] and calculate the total energy services penalty [MWh/yr]
combined_outcomes = pd.concat([chp_outcomes, w2e_outcomes, pulp_outcomes])
# combined_outcomes = combined_outcomes[ combined_outcomes["penalty_biomass"] == 0 ]

grouped = combined_outcomes.groupby("Name")["capture_cost"].mean()
grouped = combined_outcomes.groupby("Name")["captured"].mean()
sorted_grouped = grouped.sort_values(ascending=False)

data = combined_outcomes.set_index("Name").loc[sorted_grouped.index].reset_index()
data["penalty_services_total"] = data["penalty_services"] * data["captured"] /1000 # [GWh/yr]
data["penalty_biomass_total"] = data["penalty_biomass"] * data["captured"] /1000 # [GWh/yr]

# Calculate the mean, 5th percentile, and 95th percentile values
summary = data.groupby("Name").agg({
    "capture_cost": ['mean', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
    "penalty_services_total": ['mean', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
    "penalty_biomass_total": ['mean', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
    "captured": ['mean', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)] # [kt/yr]
})
summary.columns = [
    "cost_mean", "cost_5th", "cost_95th",
    "services_mean", "services_5th", "services_95th",
    "biomass_mean", "biomass_5th", "biomass_95th",
    "CO2_mean", "CO2_5th", "CO2_95th"
]
summary = summary.reindex(sorted_grouped.index)
print(summary)

# Identify the origin of each "Name" for color coding
color_map = {}
color_map.update({name: 'black' for name in chp_outcomes["Name"].unique()})
color_map.update({name: 'grey' for name in w2e_outcomes["Name"].unique()})
color_map.update({name: 'mediumseagreen' for name in pulp_outcomes["Name"].unique()})

# Truncate x-axis labels to a maximum of 17 characters
def truncate_label(label, max_length=17):
    if len(label) > max_length:
        return label[:max_length] + '.'
    return label

# Plot the mean capture cost and percentiles on the primary y-axis
fig, ax1 = plt.subplots(figsize=(15, 10))
ax1.step(summary.index, summary['cost_mean'], where='mid', label='Mean capture cost [EUR/tCO2]', marker='s', color='black', linewidth=0)
ax1.fill_between(summary.index, summary['cost_5th'], summary['cost_95th'], step='mid', color='grey', alpha=0.2, label='5th-95th percentile range')

# Perform linear regression for the mean capture cost
x = np.arange(len(summary.index))  # Numeric index for regression
y = summary['cost_mean'].values
coefficients = np.polyfit(x, y, 1)  # Linear fit
poly = np.poly1d(coefficients)
ax1.plot(summary.index, poly(x), color='black', linestyle='--', label='Regression of capture cost')

ax1.set_xlabel('Name')
ax1.set_ylabel('Capture cost [EUR/tCO2]', fontsize=14)
ax1.set_title('Capture Cost and Energy Penalties by Plant', fontsize=14)
# ax1.tick_params(axis='x', rotation=90)
labels = [truncate_label(label) for label in summary.index]
ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=10)

ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=90, ha='right')
ymin, ymax = ax1.get_ylim()
ax1.set_ylim(-40, 400)

# Add which secondary axes you want to plot
ax2 = ax1.twinx()
ax2.step(summary.index, summary['services_mean'], where='mid', label='Mean penalty energy services [GWh p.a.]', marker='s', color='crimson', linewidth=0)
ax2.fill_between(summary.index, summary['services_5th'], summary['services_95th'], step='mid', color='crimson', alpha=0.2, label='5th-95th percentile range')

ax2.step(summary.index, summary['biomass_mean'], where='mid', label='Mean penalty biomass [GWh p.a.]', marker='s', color='mediumseagreen', linewidth=0)
ax2.fill_between(summary.index, summary['biomass_5th'], summary['biomass_95th'], step='mid', color='mediumseagreen', alpha=0.2, label='5th-95th percentile range')

ax2.step(summary.index, summary['CO2_mean'], where='mid', label='Mean captured CO2 [kt p.a.]', marker='s', color='deepskyblue', linewidth=0)
ax2.fill_between(summary.index, summary['CO2_5th'], summary['CO2_95th'], step='mid', color='deepskyblue', alpha=0.2, label='5th-95th percentile range')

ymin, ymax = ax2.get_ylim()
# ax2.set_ylim(ymin + 600, ymax + 600)
ax2.set_ylim(-160, 1600)
ax2.set_ylabel('Penalty Services/Biomass', fontsize=14)
ax2.tick_params(axis='y', labelsize=12)

# Color code x-axis labels
for label in ax1.get_xticklabels():
    label.set_color(color_map.get(label.get_text(), 'black'))

# Add gridlines for every 5th name
for i in range(0, len(summary.index), 10):
    ax1.axvline(x=i, color='grey', linestyle='--', linewidth=0.5)
ax1.grid(axis='y')

# Add legends for both y-axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=14)

plt.tight_layout()
fig.savefig('MACC_all.png', dpi=600)
plt.show()