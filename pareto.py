import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline
import numpy as np
from sklearn.linear_model import LinearRegression



# Read data
chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

w2e_experiments = pd.read_csv("WASTE experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
w2e_outcomes = pd.read_csv("WASTE experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

pulp_experiments = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

# Determine example data
example = pulp_outcomes[pulp_outcomes['Name'] == "Ostrand"]

# Create vectors
X = example['penalty_services'].values
Y = example['penalty_biomass'].values
Z = example['captured'].values


# # Plotting
# plt.figure(figsize=(10, 6))
# plt.hist2d(X, Y, bins=30, cmap='Blues')
# plt.colorbar(label='Counts')

# plt.xlabel('Penalty Services')
# plt.ylabel('Penalty Biomass')
# plt.title('2D Histogram of Penalty Services and Penalty Biomass')
# plt.grid(True)

# # Scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(X, Y, alpha=0.5, edgecolor='k')
# plt.xlabel('Penalty Services')
# plt.ylabel('Penalty Biomass')
# plt.title('Scatter Plot of Penalty Services and Penalty Biomass')
# plt.grid(True)

# plt.show()

# Read data and prepare example data
example = pulp_outcomes[pulp_outcomes['Name'] == "Ostrand"]

# Create bins for penalty_biomass
num_bins = 10
example['penalty_biomass_bins'] = pd.cut(example['penalty_biomass'], bins=num_bins)

# Group by bins
grouped = example.groupby('penalty_biomass_bins')

# Lists to store bin ranges and corresponding statistics
bin_ranges = []
means = []
percentile_5th = []
percentile_95th = []

# Iterate over groups to calculate statistics and plot
for bin_range, group in grouped:
    bin_ranges.append(bin_range)
    
    mean_val = group['penalty_services'].mean()
    percentile_5th_val = group['penalty_services'].quantile(0.05)
    percentile_95th_val = group['penalty_services'].quantile(0.95)
    
    means.append(mean_val)
    percentile_5th.append(percentile_5th_val)
    percentile_95th.append(percentile_95th_val)
    
    # print(f"\nBin range: {bin_range}")
    # print(group)
    # print(f"Mean: {mean_val}, 5th percentile: {percentile_5th_val}, 95th percentile: {percentile_95th_val}")

# Convert lists to numpy arrays
bin_centers = np.array([bin.mid for bin in bin_ranges])  # Get the bin centers for plotting
bin_centers[0] = 0
means = np.array(means)
percentile_5th = np.array(percentile_5th)
percentile_95th = np.array(percentile_95th)

# Remove NaN and Inf values
valid_indices = ~np.isnan(means) & ~np.isinf(means)
bin_centers = bin_centers[valid_indices]
means = means[valid_indices]
percentile_5th = percentile_5th[valid_indices]
percentile_95th = percentile_95th[valid_indices]

# Smooth lines using spline interpolation
bin_centers_smooth = np.linspace(min(bin_centers), max(bin_centers), 300)
means_smooth = make_interp_spline(bin_centers, means, k=2)(bin_centers_smooth)
percentile_5th_smooth = make_interp_spline(bin_centers, percentile_5th, k=2)(bin_centers_smooth)
percentile_95th_smooth = make_interp_spline(bin_centers, percentile_95th, k=2)(bin_centers_smooth)

# Linear regression for means
mean_reg = LinearRegression().fit(bin_centers.reshape(-1, 1), means)
mean_line = mean_reg.predict(bin_centers.reshape(-1, 1))

# Linear regression for 5th percentile
percentile_5th_reg = LinearRegression().fit(bin_centers.reshape(-1, 1), percentile_5th)
percentile_5th_line = percentile_5th_reg.predict(bin_centers.reshape(-1, 1))

# Linear regression for 95th percentile
percentile_95th_reg = LinearRegression().fit(bin_centers.reshape(-1, 1), percentile_95th)
percentile_95th_line = percentile_95th_reg.predict(bin_centers.reshape(-1, 1))

# Plotting smooth lines
plt.figure(figsize=(10, 6))
plt.plot(bin_centers_smooth, means_smooth, color='blue', label='Mean (Smooth)')
plt.plot(bin_centers_smooth, percentile_5th_smooth, color='green', label='5th Percentile (Smooth)')
plt.plot(bin_centers_smooth, percentile_95th_smooth, color='red', label='95th Percentile (Smooth)')

# Add scatter points for means, 5th and 95th percentiles
plt.scatter(bin_centers, means, color='blue', edgecolor='k', zorder=5)
plt.scatter(bin_centers, percentile_5th, color='green', edgecolor='k', zorder=5)
plt.scatter(bin_centers, percentile_95th, color='red', edgecolor='k', zorder=5)

plt.xlabel('Penalty Biomass (Bin Centers)')
plt.ylabel('Penalty Services')
plt.title('Penalty Services Statistics by Penalty Biomass Bins (Smooth Lines)')
plt.legend()
plt.grid(True)
# plt.show()

# Plotting linear regression lines
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, mean_line, color='blue', label='Mean (Linear Regression)')
plt.plot(bin_centers, percentile_5th_line, color='green', label='5th Percentile (Linear Regression)')
plt.plot(bin_centers, percentile_95th_line, color='red', label='95th Percentile (Linear Regression)')

# Add scatter points for means, 5th and 95th percentiles
plt.scatter(bin_centers, means, color='blue', edgecolor='k', zorder=5)
plt.scatter(bin_centers, percentile_5th, color='green', edgecolor='k', zorder=5)
plt.scatter(bin_centers, percentile_95th, color='red', edgecolor='k', zorder=5)

plt.xlabel('Penalty Biomass (Bin Centers)')
plt.ylabel('Penalty Services')
plt.title('Penalty Services Statistics by Penalty Biomass Bins (Linear Regression)')
plt.legend()
plt.grid(True)
plt.show()