import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def filter_dataframes(experiments_df, outcomes_df, conditions, categorical_conditions=None):

    # Build the query string for numerical conditions
    query = []
    for column, (min_value, max_value) in conditions.items():
        if min_value is not None and max_value is not None:
            query.append(f"{column} >= {min_value} & {column} <= {max_value}")
        elif min_value is not None:
            query.append(f"{column} >= {min_value}")
        elif max_value is not None:
            query.append(f"{column} <= {max_value}")
    
    # Apply numerical conditions
    if query:
        query_string = " & ".join(query)
        print("Query String:", query_string)  # Debugging: Print the query string
        filtered_experiments_df = experiments_df.query(query_string)
    else:
        filtered_experiments_df = experiments_df

    # Apply categorical conditions
    if categorical_conditions:
        for column, allowed_values in categorical_conditions.items():
            if column in filtered_experiments_df.columns:
                filtered_experiments_df = filtered_experiments_df[filtered_experiments_df[column].isin(allowed_values)]
            else:
                print(f"Column '{column}' not found in experiments dataframe.")  # Debugging

    # Filter outcomes dataframe based on the filtered experiments dataframe
    filtered_outcomes_df = outcomes_df.loc[filtered_experiments_df.index.intersection(outcomes_df.index)]

    # Print information about filtering results
    if filtered_experiments_df.empty:
        print("Filtered dataframe is empty.")
    else:
        removed_count = len(experiments_df) - len(filtered_experiments_df)
        removed_percent = (removed_count / len(experiments_df)) * 100
        print(f"Removed {removed_count} rows, or {removed_percent:.2f}% of total scenarios.")

    return filtered_experiments_df, filtered_outcomes_df

# NOTE: WE APPLY VARIOUS FILTERS TO PICK ONLY A SUBSET WHICH IS PROMISING FOR LOW ENERGY PENALTIES
# FILTER CHP DATA
chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')
grouped = chp_outcomes.groupby('Name')
gross_means = grouped['gross'].mean().sort_values()

high_gross_names = gross_means[(gross_means > 300)].index.tolist()
boolean = chp_outcomes['Name'].isin(high_gross_names)
filtered_outcomes_high = chp_outcomes[boolean].reset_index(drop=True)
filtered_experiments_high = chp_experiments[boolean].reset_index(drop=True)

mid_gross_names = gross_means[(gross_means > 200) & (gross_means < 300)].index.tolist()
boolean = chp_outcomes['Name'].isin(mid_gross_names)
filtered_outcomes_mid = chp_outcomes[boolean].reset_index(drop=True)
filtered_experiments_mid = chp_experiments[boolean].reset_index(drop=True)

low_gross_names = gross_means[(gross_means < 200)].index.tolist()
boolean = chp_outcomes['Name'].isin(low_gross_names)
filtered_outcomes_low = chp_outcomes[boolean].reset_index(drop=True)
filtered_experiments_low = chp_experiments[boolean].reset_index(drop=True)

numerical_restrictions_1 = {
    'COP': (3.14, 3.80),
    'celc': (20, 48),
    # # 'rate': (0.78, 0.893),
    # # 'i': (0.05, 0.10),
    # 'time': (5305, 5999),
    # "duration_increase": (None, 1001)
}
categorical_restrictions_1 = {
    "heat_pump": [True],
    "duration_increase": [0]
}
numerical_restrictions_2 = {
    # 'COP': (2.45, 3.80),
    'celc': (20, 67),
    # 'rate': (0.78, 0.893),
    # 'i': (0.05, 0.077),
    'time': (5467, 5999),
    # "duration_increase": (None, 1001)
}
categorical_restrictions_2 = {
    "heat_pump": [True],
    "duration_increase": [0]
}
numerical_restrictions_3 = {
    # 'COP': (2.45, 3.80),
    # 'Tsupp': (83, 100),
    # 'rate': (0.78, 0.893),
    'i': (0.05, 0.077),
    'time': (5375, 5999),
    # "duration_increase": (None, 1001)
}
categorical_restrictions_3 = {
    "heat_pump": [True],
    "duration_increase": [0]
}


subsets = [[filtered_experiments_high,filtered_outcomes_high,numerical_restrictions_1,categorical_restrictions_1],
           [filtered_experiments_mid,filtered_outcomes_mid,numerical_restrictions_2,categorical_restrictions_2],
           [filtered_experiments_low,filtered_outcomes_low,numerical_restrictions_3,categorical_restrictions_3]]

filtered_chp_experiments = pd.DataFrame()
filtered_chp_outcomes = pd.DataFrame()

for subset in subsets:
    experiments, outcomes = filter_dataframes(subset[0], subset[1], subset[2], subset[3]) # Filters the low, mid and high subsets of CHPs
    filtered_chp_experiments = pd.concat([filtered_chp_experiments, experiments]) # Stores the filtered experiments and outcomes
    filtered_chp_outcomes = pd.concat([filtered_chp_outcomes, outcomes])

# FILTER WASTE DATA
w2e_experiments = pd.read_csv("WASTE experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
w2e_outcomes = pd.read_csv("WASTE experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')
grouped = w2e_outcomes.groupby('Name')
gross_means = grouped['gross'].mean().sort_values()

high_gross_names = gross_means[(gross_means > 350)].index.tolist()
boolean = w2e_outcomes['Name'].isin(high_gross_names)
filtered_outcomes_high = w2e_outcomes[boolean].reset_index(drop=True)
filtered_experiments_high = w2e_experiments[boolean].reset_index(drop=True)

mid_gross_names = gross_means[(gross_means > 150) & (gross_means < 350)].index.tolist()
boolean = w2e_outcomes['Name'].isin(mid_gross_names)
filtered_outcomes_mid = w2e_outcomes[boolean].reset_index(drop=True)
filtered_experiments_mid = w2e_experiments[boolean].reset_index(drop=True)

low_gross_names = gross_means[(gross_means < 150)].index.tolist()
boolean = w2e_outcomes['Name'].isin(low_gross_names)
filtered_outcomes_low = w2e_outcomes[boolean].reset_index(drop=True)
filtered_experiments_low = w2e_experiments[boolean].reset_index(drop=True)


numerical_restrictions_1 = {
    # 'COP': (3, 3.80),
    'celc': (20, 54),
    # # 'rate': (0.78, 0.893),
    'i': (0.05, 0.08),
    # 'time': (4400, 5999),
    # "duration_increase": (None, 1001)
}
categorical_restrictions_1 = {
    "heat_pump": [True],
    # "duration_increase": [0]
}
numerical_restrictions_2 = {
    # 'COP': (3, 3.80),
    'celc': (20, 64),
    # 'rate': (0.78, 0.893),
    'i': (0.05, 0.074),
    # 'time': (4822, 5999),
    # "duration_increase": (None, 1001)
}
categorical_restrictions_2 = {
    "heat_pump": [True],
    # "duration_increase": [0]
}
numerical_restrictions_3 = {
    'celc': (20, 68),
    # 'Tsupp': (83, 100),
    # 'rate': (0.78, 0.893),
    'i': (0.05, 0.072),
    # 'time': (4200, 5999),
    # "duration_increase": (None, 1001)
}
categorical_restrictions_3 = {
    "heat_pump": [True],
    # "duration_increase": [0]
}

subsets = [[filtered_experiments_high,filtered_outcomes_high,numerical_restrictions_1,categorical_restrictions_1],
           [filtered_experiments_mid,filtered_outcomes_mid,numerical_restrictions_2,categorical_restrictions_2],
           [filtered_experiments_low,filtered_outcomes_low,numerical_restrictions_3,categorical_restrictions_3]]

filtered_w2e_experiments = pd.DataFrame()
filtered_w2e_outcomes = pd.DataFrame()

for subset in subsets:
    experiments, outcomes = filter_dataframes(subset[0], subset[1], subset[2], subset[3]) # Filters the low, mid and high subsets of CHPs
    filtered_w2e_experiments = pd.concat([filtered_w2e_experiments, experiments]) # Stores the filtered experiments and outcomes
    filtered_w2e_outcomes = pd.concat([filtered_w2e_outcomes, outcomes])

# FILTER PULP DATA
pulp_experiments = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')
pulp_coordinates = pd.read_csv('pulp_coordinates.csv')
numerical_restrictions_1 = {
    'beta': (0.6, 0.65),
    'celc': (20, 74),
    # 'rate': (0.86, 0.93),
}
categorical_restrictions_1 = {
    "SupplyStrategy": ["SteamLP", "SteamHP"],
    "BarkIncrease": [0]
}

numerical_restrictions_2 = {
    'beta': (0.6, 0.67),
    'celc': (20, 51),
    # 'BarkIncrease': (None, 31),
}
categorical_restrictions_2 = {
    "SupplyStrategy": ["SteamLP"],
    "BarkIncrease": [0]
}

numerical_restrictions_3 = {
    'beta': (0.6, 0.64),
    'celc': (20, 80),
    # 'BarkIncrease': (None, 31),
}
categorical_restrictions_3 = {
    "SupplyStrategy": ["SteamLP"],
    "BarkIncrease": [0]
}

numerical_restrictions_4 = {
    'i': (0.05, 0.095),
    'celc': (20, 85),
    # 'BarkIncrease': (None, 31),
}
categorical_restrictions_4 = {
    "SupplyStrategy": ["HeatPumps"],
    "BarkIncrease": [0]
}

numerical_restrictions_5 = {
    'COP': (3.2, 3.8),
    'celc': (20, 54),
    # 'BarkIncrease': (None, 61),
}
categorical_restrictions_5 = {
    "SupplyStrategy": ["HeatPumps"],
    "BarkIncrease": [0]
}

numerical_restrictions_6 = {
    'beta': (0.6, 0.66),
    'celc': (20, 48),
    # 'BarkIncrease': (None, 31),
}
categorical_restrictions_6 = {
    "SupplyStrategy": ["SteamLP"],
    "BarkIncrease": [0]
}

numerical_restrictions_7 = {
    'rate': (0.6, 0.66),
    'celc': (20, 62),
    # 'BarkIncrease': (None, 31),
}
categorical_restrictions_7 = {
    "SupplyStrategy": ["SteamLP"],
    "BarkIncrease": [0]
}

restrictions = [            [numerical_restrictions_1, categorical_restrictions_1], 
                            [numerical_restrictions_2, categorical_restrictions_2],
                            [numerical_restrictions_3, categorical_restrictions_3],
                            [numerical_restrictions_4, categorical_restrictions_4],
                            [numerical_restrictions_5, categorical_restrictions_5],
                            [numerical_restrictions_6, categorical_restrictions_6],
                            [numerical_restrictions_6, categorical_restrictions_6]
                            ]
separate_mills = [] # This is used to group the pulp_data into subsets, one per mill
for idx, Name in enumerate(pulp_coordinates["Name"]):
    pulp_experiments_subset = pulp_experiments[pulp_experiments["Name"] == Name].reset_index(drop=True)
    # print(pulp_experiments_subset) # This should print correctly for all iterations
    pulp_outcomes_subset = pulp_outcomes[pulp_outcomes["Name"] == Name].reset_index(drop=True)
    # print(restrictions[idx][0])
    separate_mills.append([pulp_experiments_subset, pulp_outcomes_subset, restrictions[idx][0], restrictions[idx][1]])

filtered_pulp_experiments = pd.DataFrame()
filtered_pulp_outcomes = pd.DataFrame()

for subset in separate_mills:
    experiments, outcomes = filter_dataframes(subset[0], subset[1], subset[2], subset[3])
    # print(outcomes)
    filtered_pulp_experiments = pd.concat([filtered_pulp_experiments, experiments]) # Stores the filtered experiments and outcomes
    filtered_pulp_outcomes = pd.concat([filtered_pulp_outcomes, outcomes])

# Add labels to the experiments
filtered_chp_experiments["type"] = "chp"
filtered_w2e_experiments["type"] = "w2e"
filtered_pulp_experiments["type"] = "pulp"

filtered_chp_outcomes["type"] = "chp"
filtered_w2e_outcomes["type"] = "w2e"
filtered_pulp_outcomes["type"] = "pulp"

# COMBINE DATAFRAMES and find sorting order
combined_experiments = pd.concat([filtered_chp_experiments, filtered_w2e_experiments, filtered_pulp_experiments])
combined_outcomes = pd.concat([filtered_chp_outcomes, filtered_w2e_outcomes, filtered_pulp_outcomes])

grouped = combined_outcomes.groupby("Name")["penalty_services"].mean()
sorted_grouped = grouped.sort_values(ascending=True)

data = combined_outcomes.set_index("Name").loc[sorted_grouped.index].reset_index()
data["penalty_services_total"] = data["penalty_services"] * data["captured"] /1000 # [GWh/yr]
# data["penalty_biomass_total"] = data["penalty_biomass"] * data["captured"] /1000 # [GWh/yr]

# Calculate the mean, 5th percentile, and 95th percentile values
summary = data.groupby("Name").agg({
    "capture_cost": ['mean', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
    "penalty_services": ['mean', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
    "penalty_services_total": ['mean', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
    "captured": ['mean', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],  # [kt/yr]
    "type": lambda x: x.iloc[0]  # Take the first type (since all rows of the same 'Name' have the same 'type')
})

summary.columns = [
    "cost_mean", "cost_5th", "cost_95th",
    "services_mean", "services_5th", "services_95th",
    "total_mean", "total_5th", "total_95th",
    "CO2_mean", "CO2_5th", "CO2_95th",
    "type"  # Add type as the last column
]

# Ensure the summary follows the sorted order of 'Name'
summary = summary.reindex(sorted_grouped.index)


# Normalize the color map based on the cost_mean for better visualization
norm = plt.Normalize(0, sum(summary['total_95th'])) #NOTE: Color based on cumulative energy penalty instead!
cmap = plt.get_cmap("viridis")

def truncate_label(label, max_length=17):
    if len(label) > max_length:
        return label[:max_length] + '.'
    return label

# Initialize the plot
fig, ax = plt.subplots(figsize=(14, 8))
ax2 = ax.twinx() 

# Variables to store cumulative x and y values for the connected cumulative penalty
x_values = []
y_values = []

# Plot each box for each Name
x_start = 0
x_positions = []
color_5th = 0
color_mean = 0
color_95th = 0
target_count = 0
for i, name in enumerate(summary.index):
    services_5th = summary.loc[name, "services_5th"]
    services_mean = summary.loc[name, "services_mean"]
    services_95th = summary.loc[name, "services_95th"]
    CO2_mean = summary.loc[name, "CO2_mean"]
    cost_mean = summary.loc[name, "cost_mean"]

    total_5th = summary.loc[name, "total_5th"]
    total_mean = summary.loc[name, "total_mean"]
    total_95th = summary.loc[name, "total_95th"]
    color_5th += total_5th
    color_mean += total_mean
    color_95th += total_95th

    type_i = summary.loc[name, "type"]
    if type_i == "chp":
        type_color = "black"
    elif type_i == "w2e":
        type_color = "black"
    else:
        type_color = "black"

    x_end = x_start + CO2_mean

    # Create a rectangle for each Name 1800 kt (2030), 3000-10000 (2045)
    rect = plt.Rectangle(
        (x_start, services_5th),  # Bottom-left corner
        CO2_mean,  # Width
        services_95th - services_5th,  # Height
        color="grey",
        alpha=0.4,
        edgecolor=None
    )
    ax.add_patch(rect)
    # ax.plot([x_start, x_end], [services_5th, services_5th], color=cmap(norm(color_5th)), linewidth=2, alpha=0.4)
    # ax.plot([x_start, x_end], [services_mean, services_mean], color=cmap(norm(color_mean)), linewidth=4)

    ax.plot([x_start, x_end], [services_mean, services_mean], color=type_color, linewidth=3)
    # ax.plot([x_start, x_end], [services_95th, services_95th], color=cmap(norm(color_95th)), linewidth=2, alpha=0.4)

    # Store the x and y values for the cumulative total_mean
    x_values.extend([x_start, x_end])  # Add both the starting and ending x values
    y_values.extend([color_mean, color_mean])  # Add both the starting and ending y values
    # ax2.plot([x_start, x_end], [color_mean, color_mean], color="red", linewidth=2, linestyle="--")

    # Store the x position for the label
    x_positions.append(x_end)

    if x_end > 1800 and target_count < 1:
        print("Cumulative ES penalty in 2030 [GWh/yr] =", color_5th)
        print("Cumulative ES penalty in 2030 [GWh/yr] =", color_mean)
        print("Cumulative ES penalty in 2030 [GWh/yr] =", color_95th)
        target_count += 1

    if x_end > 10000 and target_count < 2:
        print("Cumulative ES penalty in 2045 [GWh/yr] =", color_5th)
        print("Cumulative ES penalty in 2045 [GWh/yr] =", color_mean)
        print("Cumulative ES penalty in 2045 [GWh/yr] =", color_95th)
        target_count += 1

    if x_end > 19000 and target_count < 3:
        print("Cumulative ES penalty @>20 Mt [GWh/yr] =", color_5th)
        print("Cumulative ES penalty @>20 Mt [GWh/yr] =", color_mean)
        print("Cumulative ES penalty @>20 Mt [GWh/yr] =", color_95th)
        print("Total CO2 captured across all plants [Mt/yr] =", x_end/1000)
        target_count += 1

    x_start = x_end

# Plot the connected dashed line on the secondary axis
ax2.plot(x_values, y_values, color="crimson", linewidth=3, linestyle="-")

# # Add a color bar to show the cost_mean scale
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('Cumulative energy services penalty [GWh/yr]', fontsize=14)
# cbar.ax.tick_params(labelsize=12)

# Add vertical dashed lines with annotations
ax.axvline(x=1800, color='deepskyblue', linestyle='--', linewidth=3, alpha=0.8)
ax.axvline(x=10000, color='deepskyblue', linestyle='--', linewidth=3, alpha=0.8)

# Annotate the lines
ax.text(1800+100, ax.get_ylim()[1] * 0.90, '2030 target', color='deepskyblue', fontsize=14, ha='left', va='bottom')
ax.text(10000+100, ax.get_ylim()[1] * 0.90, '2045 target (indicative)', color='deepskyblue', fontsize=14, ha='left', va='bottom')

# Set the limits of the plot
ax.set_xlim(0, x_start)
ax.set_ylim(summary["services_5th"].min(), summary["services_95th"].max()+50)

# # Add evenly spaced x-axis ticks and labels
# xticks = np.linspace(0, round(x_start), num=10)
# ax.set_xticks(xticks)
# ax.set_xticklabels([f'{int(x):,}' for x in xticks], fontsize=12)

# Add light grids
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.9)

# Customize the plot
ax.set_xlabel("Cumulative captured CO2 [kt p.a.]", fontsize=14)
ax.set_ylabel("Energy services penalty [kWh/tCO2]", fontsize=14)
ax.set_title("MACC of energy services penalty", fontsize=14)
ax2.set_ylabel('Cumulative energy services penalty [GWh p.a.]', fontsize=14, color="crimson")

ax2.tick_params(axis='y', colors='crimson')  # Color the y-axis ticks and labels red
ax2.yaxis.label.set_color('crimson')  # Set the label color to red

fig.savefig('MACC_penalty.png', dpi=600)
plt.show()