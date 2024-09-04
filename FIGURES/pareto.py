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

# NOTE: Here I should filter into PRIMED data!

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

results_demand_list = []
for desired_capture in [25000]: # [kt/yr], consider perhaps only CHPs for 5000?
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

        # print(f"\nCalculating demands for {name}:")
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
    results_demand_list.append(results_demand)

# TIME FOR PLOTTING
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

# Create each frame and filter outliers
all_captured = []
for result in results_demand_list:
    filtered_result = result
    # filtered_result = result[result["electricity_5th"] > 0]
    all_captured.extend(filtered_result["captured"])
all_captured = np.array(all_captured)

norm = plt.Normalize(vmin=min(all_captured), vmax=max(all_captured))
cmap = plt.cm.viridis


fig, ax1 = plt.subplots(figsize=(16, 6))

for i, results_demand in enumerate(results_demand_list):
    results_filtered = results_demand
    # results_filtered = results_demand[results_demand["electricity_5th"] > 0]

    # Scatter plot for mean and percentile electricity
    scatter1 = ax1.scatter(results_filtered["biomass"], results_filtered["electricity"], c="black", label=f'Mean Penalty', marker='s')
    # scatter2 = ax1.scatter(results_filtered["biomass"], results_filtered["electricity_5th"], c="grey", label=f'5th Percentile Penalty', marker='v')
    # scatter3 = ax1.scatter(results_filtered["biomass"], results_filtered["electricity_95th"], c="grey", label=f'95th Percentile Penalty', marker='^')

    # Plot quadratic regression curves
    plot_quadratic_regression(results_filtered["biomass"], results_filtered["electricity"], f'Mean Penalty (regression)', '-', ax1, color="black")
    plot_quadratic_regression(results_filtered["biomass"], results_filtered["electricity_5th"], f'5th Percentile Penalty (regression)', '--', ax1, color="grey")
    plot_quadratic_regression(results_filtered["biomass"], results_filtered["electricity_95th"], f'95th Percentile Penalty (regression)', '--', ax1, color="grey")

    # Create a secondary y-axis for captured values
    mean_captured_color = plt.cm.viridis(norm(results_filtered["captured"].mean()))
    ax2 = ax1.twinx()
    sc = ax2.scatter(results_filtered["biomass"], results_filtered["captured"], c=results_filtered["captured"], cmap=cmap, label="Total Captured CO2")
    plot_quadratic_regression(results_filtered["biomass"], results_filtered["captured"], f'Total Captured CO2 (regression)', '-', ax2, color=mean_captured_color, degree=1)
    ax2.set_ylim(0, ax2.get_ylim()[1])  # Ensure the secondary y-axis starts from zero

    # # Annotate each point with its index
    # for bini, (biomass, electricity) in enumerate(zip(results_demand["biomass"], results_demand["electricity"])):
    #     ax1.annotate(str(bini), (biomass, electricity), textcoords="offset points", xytext=(random.randint(0,10),random.randint(0,10)), ha='center', c=results_filtered["captured"], cmap=cmap)


# Adding color bar to indicate the 'captured' values
cbar = plt.colorbar(sc, ax=ax1)
cbar.set_label('Captured CO2 [kt p.a.]')
cbar.ax.yaxis.set_ticks_position('right')
cbar.ax.yaxis.set_label_position('right')

# Adding plot details for ax1
ax1.set_title('Biomass vs. Energy Services Penalties')
ax1.set_xlabel('Biomass Demand [MWh p.a.]')
ax1.set_ylabel('Electricity/District Heating Penalties [MWh p.a.]')
ax1.legend(loc='lower left')
ax1.grid(True)

# Adding plot details for ax2
ax2.set_ylabel('Captured')
ax2.legend(loc='upper right')

plt.show()