import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

if 'penalty_biomass' not in chp_outcomes.columns or 'Name' not in chp_outcomes.columns:
    raise ValueError("Required columns 'penalty_biomass' or 'Name' not found in chp_outcomes DataFrame.")

chp_outcomes = chp_outcomes.reset_index()

def binning_biomass(group):
    zero_values = group[group['penalty_biomass'] == 0]
    non_zero_values = group[group['penalty_biomass'] != 0]
    
    bins = pd.cut(non_zero_values['penalty_biomass'], bins=9)
    non_zero_values['biomass_bins'] = bins
    
    zero_values['biomass_bins'] = pd.Interval(left=-float('inf'), right=float('inf'), closed='right')
    combined = pd.concat([zero_values, non_zero_values])
    
    return combined

# Apply the binning function to each group
chp_outcomes = chp_outcomes.groupby('Name').apply(binning_biomass)
chp_outcomes = chp_outcomes.reset_index(drop=True)

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
print("\nStatistics for each biomass bin:")
print(stats_df)

print("\nTotal Mean Captured for each 'Name':")
print(total_captured)

# Define a function to calculate the midpoint of a bin range
def get_bin_midpoint(bin_interval):
    if bin_interval.left == -float('inf') or bin_interval.right == float('inf'):
        return 0  # Handle open-ended bins as needed
    return (bin_interval.left + bin_interval.right) / 2
stats_df['Mean Biomass'] = stats_df['Biomass Bin'].apply(get_bin_midpoint)

# Time to summarize bins. Do it by estimating the captured volumes first, so we know what plants to summarize.
name_list = [] 
estimated_capture = 0
i = 0
sorted_total_captured = total_captured.sort_values(by='Total Captured', ascending=False)

while estimated_capture < 1000 and i < len(sorted_total_captured):
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
    
    # Calculate biomass_demand
    plant_demand['Biomass Demand'] = plant_demand['Mean Captured'] * plant_demand['Mean Biomass']
    plant_demand['Electricity Demand'] = plant_demand['Mean Captured'] * plant_demand['Mean Penalty'] #[MWh/yr], negative means exporting
    plant_demand['Electricity 5th'] = plant_demand['Mean Captured'] * plant_demand['5th Penalty'] 
    plant_demand['Electricity 95th'] = plant_demand['Mean Captured'] * plant_demand['95th Penalty'] 
    
    # Print results
    # print(plant_demand[['Biomass Bin', 'Mean Captured', 'Mean Biomass', 'Biomass Demand', 'Electricity Demand','Electricity 5th','Electricity 95th']])

    # ... then summarize all demands of all plants, bin-for-bin. So iterate over the 10 bins and do this.
    for i in range(0, 10):
        total_biomass[i] += plant_demand['Biomass Demand'][i]
        total_electricity[i] += plant_demand['Electricity Demand'][i]
        total_electricity_5th[i] += plant_demand['Electricity 5th'][i]
        total_electricity_95th[i] += plant_demand['Electricity 95th'][i]
        achieved_capture[i] += plant_demand['Mean Captured'][i]

results_demand = pd.DataFrame({
    "biomass": total_biomass,
    "electricity":total_electricity,
    "electricity_5th": total_electricity_5th,
    "electricity_95th": total_electricity_95th,
    "captured": achieved_capture,
})

# PLOTTING
# Function to create regression lines
def create_regression_line(x, y, degree):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly, y)
    x_range = np.linspace(x.min(), x.max(), 100)
    x_range_poly = poly.transform(x_range.reshape(-1, 1))
    y_range_pred = model.predict(x_range_poly)
    return x_range, y_range_pred

# Prepare the data
x = np.array(results_demand["biomass"])
y1 = np.array(results_demand["electricity"])
y2 = np.array(results_demand["electricity_5th"])
y3 = np.array(results_demand["electricity_95th"])
captured = np.array(results_demand["captured"])

# Calculate the color scale based on the 'captured' values
norm = plt.Normalize(vmin=min(captured), vmax=max(captured))
colors = plt.cm.viridis(norm(captured))

# Create the plot
fig, ax = plt.subplots()

# Scatter plot with colors based on 'captured'
sc = ax.scatter(x, y1, c=captured, cmap='viridis', label='Electricity')
ax.scatter(x, y2, c=captured, cmap='viridis', marker='x', label='Electricity 5th')
ax.scatter(x, y3, c=captured, cmap='viridis', marker='^', label='Electricity 95th')

# Create regression lines
x_range1, y_range_pred1 = create_regression_line(x, y1, degree=2)
x_range2, y_range_pred2 = create_regression_line(x, y2, degree=2)
x_range3, y_range_pred3 = create_regression_line(x, y3, degree=2)

# Plot regression lines
ax.plot(x_range1, y_range_pred1, color='blue', label='Electricity (Regression)')
ax.plot(x_range2, y_range_pred2, color='red', label='Electricity 5th (Regression)')
ax.plot(x_range3, y_range_pred3, color='green', label='Electricity 95th (Regression)')

# Add colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Captured')

# Labels and legend
ax.set_xlabel("Biomass")
ax.set_ylabel("Electricity")
ax.legend()
ax.set_title("Biomass vs. Electricity with Regression Lines")

plt.show()