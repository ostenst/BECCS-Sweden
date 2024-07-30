import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline
import numpy as np
from sklearn.linear_model import LinearRegression

# I want a group of groups:

# pulp_group contains 
#   pulp_plant1 which contains
#       ensure a specific subset for biomass = 0 
#       to ensure the subsets are comparable, I force them to be "0"-"100%"
#       biomass1 subset
#       biomass2 subset
#       in each biomass subset, calculate Statistics
#   and calculate plant1 mean captured emisisons

#   pulp_plant2 which contains
#       biomass1 subset
#       biomass2 subset

# Now I can summarize any pulp_planti with any chp_plantj, by adding 0-100% bins.
# What I will do is to concatenate chp,waste,pulp, and sort by captured emissions.
# Then I will try different levels (i.e. number of captured plants, save their names!), and add these together.

# Read data
chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

w2e_experiments = pd.read_csv("WASTE experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
w2e_outcomes = pd.read_csv("WASTE experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

pulp_experiments = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

# Adding all together!
chp_outcomes = pd.concat([chp_outcomes, w2e_outcomes])
chp_outcomes = pd.concat([chp_outcomes, pulp_outcomes])

if 'penalty_biomass' not in chp_outcomes.columns or 'Name' not in chp_outcomes.columns:
    raise ValueError("Required columns 'penalty_biomass' or 'Name' not found in chp_outcomes DataFrame.")

chp_outcomes = chp_outcomes.reset_index()

def binning_biomass(group):
    # I think the bins are ok now???
    zero_values = group[group['penalty_biomass'] == 0]
    non_zero_values = group[group['penalty_biomass'] != 0]
    
    if non_zero_values.empty:
        # This is True for Aspa, and for W2E plants 
        zero_values['biomass_bins'] = pd.Interval(left=-float('inf'), right=float('inf'), closed='right')
        combined = zero_values
    else:
        # Define bin edges
        bin_edges = pd.qcut(non_zero_values['penalty_biomass'], q=8, retbins=True)[1]
        bin_edges = [-float('inf')] + list(bin_edges) + [float('inf')] # WHEN THIS CREATES 8-9 BINS???
        # bin_edges = list(bin_edges) + [float('inf')] # WHY DOES THIS ONLY CREATE 2 BINS
        
        # Print bin edges for debugging
        print(f"Bin edges for group '{group['Name'].iloc[0]}': {bin_edges}")
        print("The bins are now ok...")

        # Assign bins to non-zero values
        non_zero_values['biomass_bins'] = pd.cut(non_zero_values['penalty_biomass'], bins=bin_edges, include_lowest=True)
        
        # Assign zero values to the first bin
        if not zero_values.empty:
            zero_values['biomass_bins'] = pd.Interval(left=-float('inf'), right=bin_edges[1], closed='right')
            # Could this be adding a new bin?
        
        # Combine the zero and non-zero value DataFrames
        combined = pd.concat([zero_values, non_zero_values])

    return combined
# def binning_biomass(group):
#     zero_values = group[group['penalty_biomass'] == 0]
#     non_zero_values = group[group['penalty_biomass'] != 0]
    
#     if non_zero_values.empty:
#         # If all values are zero, assign them to a single bin
#         zero_values['biomass_bins'] = pd.Interval(left=-float('inf'), right=float('inf'), closed='right')
#         combined = zero_values
#     else:
#         # Define 10 bins based on penalty_biomass
#         bin_edges = pd.qcut(non_zero_values['penalty_biomass'], q=8, retbins=True)[1]
#         bin_edges = [-float('inf')] + list(bin_edges) + [float('inf')]
        
#         # Assign bins to non-zero values
#         non_zero_values['biomass_bins'] = pd.cut(non_zero_values['penalty_biomass'], bins=bin_edges, include_lowest=True)
        
#         # Combine non-zero values with zero values
#         combined = pd.concat([non_zero_values, zero_values])
    
#     return combined

# Apply the binning function to each group
chp_outcomes = chp_outcomes.groupby('Name').apply(binning_biomass)
chp_outcomes = chp_outcomes.reset_index(drop=True)
for name, group in chp_outcomes.groupby('Name'):
    print(f"\nBin edges for plant '{name}':")
    print(group['biomass_bins'].unique())
print("The bins are now distorted!!!")

grouped = chp_outcomes.groupby(['Name', 'biomass_bins'])
total_captured = chp_outcomes.groupby('Name')['captured'].mean().reset_index()
total_captured.rename(columns={'captured': 'Total Captured'}, inplace=True)
total_captured = total_captured.sort_values(by='Total Captured', ascending=False)

# Calculate statistics within each bin
statistics = []
for (name, bin_name), group in grouped:
    mean_captured = group['captured'].mean() if 'captured' in group.columns else None
    mean_penalty_services = group['penalty_services'].mean() if 'penalty_services' in group.columns else None
    percentile_5th_val = group['penalty_services'].quantile(0.05) if 'penalty_services' in group.columns else None
    percentile_95th_val = group['penalty_services'].quantile(0.95) if 'penalty_services' in group.columns else None
    
    stats = {
        'Name': name,
        'Biomass Bin': bin_name,
        'Mean Captured': mean_captured,
        'Mean Penalty': mean_penalty_services,
        '5th Penalty': percentile_5th_val,
        '95th Penalty': percentile_95th_val
    }
    statistics.append(stats)

stats_df = pd.DataFrame(statistics)
print("\nStatistics for each biomass bin and plant:")
print(stats_df)
# print(stats_df["Name"].unique())

print("\nTotal Mean Captured for each 'Name':")
print(total_captured)

# Define a function to calculate the midpoint of a bin range
def get_bin_midpoint(bin_interval):
    if bin_interval.left == -float('inf') or bin_interval.right == float('inf'):
        return 0  # Handle open-ended bins as needed
    return (bin_interval.left + bin_interval.right) / 2
stats_df['Mean Biomass'] = stats_df['Biomass Bin'].apply(get_bin_midpoint)

# Time to summarize bins. Do it by estimating the captured volumes first, to append correct plants to the list.
name_list = [] 
estimated_capture = 0
i = 0
sorted_total_captured = total_captured.sort_values(by='Total Captured', ascending=False)

while estimated_capture < 10000 and i < len(sorted_total_captured):
    estimated_capture += sorted_total_captured["Total Captured"].iloc[i]
    name_list.append(sorted_total_captured["Name"].iloc[i])
    i += 1
print("\nName list sorted by Total Captured and ensuring estimated_capture >= 1000:")
print(name_list)

total_biomass = [0] * 10
total_electricity = [0] * 10
total_electricity_5th = [0] * 10
total_electricity_95th = [0] * 10
achieved_capture = [0] * 10
for name in name_list:
    # The logic is this: estimate all demands of 1 plant in each of its 10 biomass bins....
    print(f"\nCalculating biomass demand for {name}:")
    plant_demand = stats_df[stats_df['Name'] == name].reset_index(drop=True)
    # print(plant_demand)
    # Calculate biomass_demand
    plant_demand['Biomass Demand'] = plant_demand['Mean Captured'] * plant_demand['Mean Biomass']
    plant_demand['Electricity Demand'] = plant_demand['Mean Captured'] * plant_demand['Mean Penalty'] #[MWh/yr], negative means exporting
    plant_demand['Electricity 5th'] = plant_demand['Mean Captured'] * plant_demand['5th Penalty'] 
    plant_demand['Electricity 95th'] = plant_demand['Mean Captured'] * plant_demand['95th Penalty'] 
    
    # Print results
    # print(plant_demand[['Biomass Bin', 'Mean Captured', 'Mean Biomass', 'Biomass Demand', 'Electricity Demand','Electricity 5th','Electricity 95th']])
    
    # ... then summarize all demands of all plants, bin-for-bin. So iterate over the 10 bins and do this. Sometimes we have only 1 bin (Aspa and W2e plants)
    # Check if there's only one bin for the plant
    if len(plant_demand) == 1:
        print(" ZERO BIOMASS ")
        total_biomass[0] += plant_demand['Biomass Demand'][0] 
        total_electricity[0] += plant_demand['Electricity Demand'][0]
        total_electricity_5th[0] += plant_demand['Electricity 5th'][0]
        total_electricity_95th[0] += plant_demand['Electricity 95th'][0]
        achieved_capture[0] += plant_demand['Mean Captured'][0]
    else:
        for i in range(0,len(plant_demand)):
            total_biomass[i] += plant_demand['Biomass Demand'][i]
            total_electricity[i] += plant_demand['Electricity Demand'][i]
            total_electricity_5th[i] += plant_demand['Electricity 5th'][i]
            total_electricity_95th[i] += plant_demand['Electricity 95th'][i]
            achieved_capture[i] += plant_demand['Mean Captured'][i]
    # print(total_biomass)
    # print(total_electricity)

results_demand = pd.DataFrame({
    "biomass": total_biomass,
    "electricity":total_electricity,
    "electricity_5th": total_electricity_5th,
    "electricity_95th": total_electricity_95th,
    "captured": achieved_capture,
})

# Calculate the color scale based on the 'captured' values
norm = plt.Normalize(vmin=min(results_demand["captured"]), vmax=max(results_demand["captured"]))
colors = plt.cm.viridis(norm(results_demand["captured"]))

# Plotting the scatter plot
plt.figure(figsize=(10, 6))

# Scatter plot for mean electricity
plt.scatter(results_demand["biomass"], results_demand["electricity"], c=colors, label='Mean Electricity', marker='o')
# Scatter plot for 5th percentile electricity
plt.scatter(results_demand["biomass"], results_demand["electricity_5th"], c=colors, label='5th Percentile Electricity', marker='x')
# Scatter plot for 95th percentile electricity
plt.scatter(results_demand["biomass"], results_demand["electricity_95th"], c=colors, label='95th Percentile Electricity', marker='s')

# Adding color bar to indicate the 'captured' values
sc = plt.scatter(results_demand["biomass"], results_demand["electricity"], c=results_demand["captured"], cmap='viridis')
plt.colorbar(sc, label='Captured')

# Adding plot details
plt.title('Biomass vs. Electricity Demand')
plt.xlabel('Biomass Demand')
plt.ylabel('Electricity Demand (MWh/yr)')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
