import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline
import numpy as np
import numpy.random as random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Read data
chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

w2e_experiments = pd.read_csv("WASTE experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
w2e_outcomes = pd.read_csv("WASTE experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

pulp_experiments = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

# Adding all together!
renova_df = w2e_outcomes[ (w2e_outcomes["Name"]=="Renova ") ].iloc[0:300,:].reset_index()
chp_outcomes = pd.concat([chp_outcomes, w2e_outcomes])
chp_outcomes = pd.concat([chp_outcomes, pulp_outcomes])

if 'penalty_biomass' not in chp_outcomes.columns or 'Name' not in chp_outcomes.columns:
    raise ValueError("Required columns 'penalty_biomass' or 'Name' not found in chp_outcomes DataFrame.")

chp_outcomes = chp_outcomes.reset_index()
# chp_outcomes = chp_outcomes.iloc[300:1000,:]
# chp_outcomes = pd.concat([chp_outcomes, renova_df])
# print(chp_outcomes)

def binning_biomass(group):
    num_bins = 14
    
    # Separate the group into zero and non-zero penalty_biomass
    zero_penalty = group[group['penalty_biomass'] == 0]
    non_zero_penalty = group[group['penalty_biomass'] > 0]
    
    if not non_zero_penalty.empty:
        # Create bins for the non-zero penalty_biomass values
        bin_edges = pd.cut(non_zero_penalty['penalty_biomass'], bins=num_bins, retbins=True)[1]
        
        # Create a categorical bin column
        non_zero_penalty['biomass_bins'] = pd.cut(non_zero_penalty['penalty_biomass'], bins=bin_edges, labels=False)
        
        # Map bin intervals to edge numbers
        bin_intervals = pd.IntervalIndex.from_breaks(bin_edges)
        bin_labels = {interval: idx for idx, interval in enumerate(bin_intervals)}
        
        # Map each bin to its corresponding edge number
        non_zero_penalty['edge_number'] = non_zero_penalty['biomass_bins'].map(bin_labels)
    else:
        # Handle the case where there are no non-zero penalties
        non_zero_penalty['biomass_bins'] = pd.Categorical([])
        non_zero_penalty['edge_number'] = pd.NA
    
    # Assign a separate bin label for zero penalty_biomass values
    zero_penalty['biomass_bins'] = 'Zero Penalty'
    zero_penalty['edge_number'] = None  # No edge number for zero penalty
    
    # Concatenate the zero and non-zero groups back together
    binned_group = pd.concat([zero_penalty, non_zero_penalty])
    
    return binned_group

# Apply the binning function to each group
chp_outcomes = chp_outcomes.groupby('Name').apply(binning_biomass)
chp_outcomes = chp_outcomes.reset_index(drop=True)
# for name, group in chp_outcomes.groupby('Name'):
#     print(f"\nBins for plant '{name}':")
#     print(len(group['biomass_bins'].unique()))

grouped = chp_outcomes.groupby(['Name', 'biomass_bins'])
total_captured = chp_outcomes.groupby('Name')['captured'].mean().reset_index()
total_captured.rename(columns={'captured': 'Total Captured'}, inplace=True)
total_captured = total_captured.sort_values(by='Total Captured', ascending=False)

# Calculate statistics within each bin
statistics = []
for (name, bin_name), group in grouped:
    mean_captured = group['captured'].mean() if 'captured' in group.columns else None
    mean_penalty_biomass = group['penalty_biomass'].mean() if 'penalty_biomass' in group.columns else None
    mean_penalty_services = group['penalty_services'].mean() if 'penalty_services' in group.columns else None
    percentile_5th_val = group['penalty_services'].quantile(0.05) if 'penalty_services' in group.columns else None
    percentile_95th_val = group['penalty_services'].quantile(0.95) if 'penalty_services' in group.columns else None
    
    stats = {
        'Name': name,
        'Biomass Bin': bin_name,
        'Mean Biomass': mean_penalty_biomass,
        'Mean Captured': mean_captured,
        'Mean Penalty': mean_penalty_services,
        '5th Penalty': percentile_5th_val,
        '95th Penalty': percentile_95th_val
    }
    statistics.append(stats)

stats_df = pd.DataFrame(statistics)
print("\nStatistics for each biomass bin and plant:")
print(stats_df)

# Decide what plants to calculate by estimating their capture potential
print(total_captured)
desired_capture = 10000 # [kt/yr]
name_list = [] 
estimated_capture = 0
i = 0

sorted_total_captured = total_captured.sort_values(by='Total Captured', ascending=False)
while estimated_capture < desired_capture and i < len(sorted_total_captured):
    estimated_capture += sorted_total_captured["Total Captured"].iloc[i]
    name_list.append(sorted_total_captured["Name"].iloc[i])
    i += 1
print("\nPlants to evaluate since cumulative estimated_capture >=", desired_capture,"kt/yr:")
print(name_list)

# Summarize demands across all bins
total_biomass = [0] * 15
total_electricity = [0] * 15
total_electricity_5th = [0] * 15
total_electricity_95th = [0] * 15
achieved_capture = [0] * 15

for name in name_list:

    print(f"\nCalculating demands for {name}:")
    plant_demand = stats_df[stats_df['Name'] == name].reset_index(drop=True)
    
    plant_demand['Biomass Demand'] = plant_demand['Mean Captured'] * plant_demand['Mean Biomass']
    plant_demand['Electricity Demand'] = plant_demand['Mean Captured'] * plant_demand['Mean Penalty'] #[MWh/yr], negative means exporting
    plant_demand['Electricity 5th'] = plant_demand['Mean Captured'] * plant_demand['5th Penalty'] 
    plant_demand['Electricity 95th'] = plant_demand['Mean Captured'] * plant_demand['95th Penalty'] 
    
    # print(plant_demand)
    
    for i, row in plant_demand.iterrows():

        if row["Biomass Bin"] == "Zero Penalty":
            total_biomass[0] += row['Biomass Demand'] #Should be zero!
            total_electricity[0] += row['Electricity Demand']
            total_electricity_5th[0] += row['Electricity 5th']
            total_electricity_95th[0] += row['Electricity 95th']
            achieved_capture[0] += row['Mean Captured']
        else:
            total_biomass[row["Biomass Bin"]+1] += row['Biomass Demand']
            total_electricity[row["Biomass Bin"]+1] += row['Electricity Demand']
            total_electricity_5th[row["Biomass Bin"]+1] += row['Electricity 5th']
            total_electricity_95th[row["Biomass Bin"]+1] += row['Electricity 95th']
            achieved_capture[row["Biomass Bin"]+1] += row['Mean Captured']
        # print(total_biomass)

results_demand = pd.DataFrame({
    "biomass": total_biomass,
    "electricity":total_electricity,
    "electricity_5th": total_electricity_5th,
    "electricity_95th": total_electricity_95th,
    "captured": achieved_capture,
})


# Create each frame and filter it.
# Create a df results_filtered which only has datapoints where total_electricity > 0
results_filtered = results_demand[results_demand["electricity"] > 0]
results_filtered = results_demand[results_demand["electricity_5th"] > 0] # NOTE: These removes some CHP populated bins, where PULP is lacking.

# Then calculate colors
all_captured = np.concatenate([results_demand1["captured"], results_demand2["captured"], results_demand3["captured"]])
norm = plt.Normalize(vmin=min(all_captured), vmax=max(all_captured))
cmap = plt.cm.viridis

# Define a function to fit and plot polynomial regression
def plot_quadratic_regression(x, y, label, linestyle, ax, color="red", degree=2):
    x = np.array(x)
    y = np.array(y)
    polynomial_features = PolynomialFeatures(degree=degree)
    model = make_pipeline(polynomial_features, LinearRegression())
    model.fit(x[:, np.newaxis], y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = model.predict(x_fit[:, np.newaxis])
    ax.plot(x_fit, y_fit, linestyle, label=label, color=color)

# Calculate the color scale based on the 'captured' values
norm = plt.Normalize(vmin=min(results_filtered["captured"]), vmax=max(results_filtered["captured"]))
colors = plt.cm.viridis(norm(results_filtered["captured"]))
mean_captured_color = plt.cm.viridis(norm(results_filtered["captured"].mean()))

fig, ax1 = plt.subplots(figsize=(16, 6))

scatter1 = ax1.scatter(results_filtered["biomass"], results_filtered["electricity"], c="black", label='Mean Electricity', marker='o')
scatter2 = ax1.scatter(results_filtered["biomass"], results_filtered["electricity_5th"], c="grey", label='5th Percentile Electricity', marker='x')
scatter3 = ax1.scatter(results_filtered["biomass"], results_filtered["electricity_95th"], c="grey", label='95th Percentile Electricity', marker='s')

# Fit and plot polynomial regression curves
plot_quadratic_regression(results_filtered["biomass"], results_filtered["electricity"], 'Mean Electricity Fit', '-', ax1, color="black")
plot_quadratic_regression(results_filtered["biomass"], results_filtered["electricity_5th"], '5th Percentile Electricity Fit', '--', ax1, color="grey")
plot_quadratic_regression(results_filtered["biomass"], results_filtered["electricity_95th"], '95th Percentile Electricity Fit', '--', ax1, color="grey")

# Adding plot details for ax1
ax1.set_title('Biomass vs. Electricity Demand')
ax1.set_xlabel('Biomass Demand')
ax1.set_ylabel('Electricity Demand (MWh/yr)')
ax1.legend()
ax1.grid(True)

# Create a secondary y-axis for captured values
ax2 = ax1.twinx()
ax2.set_ylabel('Captured')

# Scatter plot for captured values
sc = ax2.scatter(results_filtered["biomass"], results_filtered["captured"], c=results_filtered["captured"], cmap='viridis')
# Add color bar to indicate the 'captured' values
cbar = plt.colorbar(sc, ax=ax1)
cbar.set_label('Captured')
cbar.ax.yaxis.set_ticks_position('right')
cbar.ax.yaxis.set_label_position('right')

ax2.set_ylim(0, ax2.get_ylim()[1])

# Fit and plot polynomial regression curve for captured values
plot_quadratic_regression(results_filtered["biomass"], results_filtered["captured"], 'Captured Fit', '-', ax2, color=mean_captured_color, degree=1)

# Display the plot
plt.show()

# ##  BELOW WORKS BUT HAS IFFY COLORS
# # Calculate the color scale based on the 'captured' values
# norm = plt.Normalize(vmin=min(results_demand["captured"]), vmax=max(results_demand["captured"]))
# colors = plt.cm.viridis(norm(results_demand["captured"]))

# # Create a df results_filtered which only has datapoints where total_electricity > 0
# results_filtered = results_demand[results_demand["electricity"] > 0]
# results_filtered = results_demand[results_demand["electricity_5th"] > 0] # NOTE: These removes some CHP populated bins, where PULP is lacking.
# colors = plt.cm.viridis(norm(results_filtered["captured"]))

# def plot_quadratic_regression(x, y, label, linestyle, ax):
#     x = np.array(x)
#     y = np.array(y)
#     polynomial_features = PolynomialFeatures(degree=2)
#     model = make_pipeline(polynomial_features, LinearRegression())
#     model.fit(x[:, np.newaxis], y)
#     x_fit = np.linspace(x.min(), x.max(), 100)
#     y_fit = model.predict(x_fit[:, np.newaxis])
#     ax.plot(x_fit, y_fit, linestyle, label=label)

# fig, ax1 = plt.subplots(figsize=(12, 6))
# scatter1 = ax1.scatter(results_filtered["biomass"], results_filtered["electricity"], c=colors, label='Mean Electricity', marker='o')
# scatter2 = ax1.scatter(results_filtered["biomass"], results_filtered["electricity_5th"], c=colors, label='5th Percentile Electricity', marker='x')
# scatter3 = ax1.scatter(results_filtered["biomass"], results_filtered["electricity_95th"], c=colors, label='95th Percentile Electricity', marker='s')

# # Adding color bar to indicate the 'captured' values
# sc = ax1.scatter(results_filtered["biomass"], results_filtered["electricity"], c=results_filtered["captured"], cmap='viridis')
# cbar = plt.colorbar(sc, ax=ax1)
# cbar.set_label('Captured')

# # Move the colorbar to the right of the plot
# cbar.ax.yaxis.set_ticks_position('right')
# cbar.ax.yaxis.set_label_position('right')

# # Fit and plot polynomial regression curves
# plot_quadratic_regression(results_filtered["biomass"], results_filtered["electricity"], 'Mean Electricity Fit', '-', ax1)
# plot_quadratic_regression(results_filtered["biomass"], results_filtered["electricity_5th"], '5th Percentile Electricity Fit', '--', ax1)
# plot_quadratic_regression(results_filtered["biomass"], results_filtered["electricity_95th"], '95th Percentile Electricity Fit', '--', ax1)

# # # Annotate each point with its index
# # for i, (biomass, electricity) in enumerate(zip(results_demand["biomass"], results_demand["electricity"])):
# #     ax1.annotate(str(i), (biomass, electricity), textcoords="offset points", xytext=(0,10), ha='center')

# # Adding plot details for ax1
# ax1.set_title('Biomass vs. Electricity Demand')
# ax1.set_xlabel('Biomass Demand')
# ax1.set_ylabel('Electricity Demand (MWh/yr)')
# ax1.legend()
# ax1.grid(True)

# # # Create a secondary y-axis for captured values
# # ax2 = ax1.twinx()
# # ax2.plot(results_filtered["biomass"], results_filtered["captured"], 'r-', label='Captured')
# # ax2.set_ylim(0, ax2.get_ylim()[1])  # Ensure the secondary y-axis starts from zero
# # plot_quadratic_regression(results_filtered["biomass"], results_filtered["captured"], 'Captured Fit', '-', ax2)
# # ax2.set_ylabel('Captured')
# # ax2.legend(loc='upper right')

# # Display the plot
# plt.show()