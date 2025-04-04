# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# # Sample dataset
# data = pd.DataFrame({
#     'category': ['A', 'B', 'A', 'B', 'C', 'B'],
#     'price': [40, 60, 30, 80, 20, 55],
#     'quality': [7, 8, 6, 9, 5, 7]
# })

# # Convert category column to numeric levels
# category_mapping = {cat: i for i, cat in enumerate(sorted(data['category'].unique()))}
# data['category_numeric'] = data['category'].map(category_mapping)

# # Normalize numerical columns
# scaler = MinMaxScaler()
# data_scaled = pd.DataFrame(scaler.fit_transform(data[['price', 'quality']]), columns=['price', 'quality'])

# # Normalize category values to match numerical scale
# min_cat, max_cat = min(category_mapping.values()), max(category_mapping.values())
# data_scaled['category'] = (data['category_numeric'] - min_cat) / (max_cat - min_cat)  # Scale to [0,1]

# # Define a function to set colors based on conditions
# def get_color(row):
#     if row['category'] == 1 and row['price'] > 50:  # Category B and price > 50
#         return 'red'  
#     elif row['category'] == 1:  # Other category B items
#         return 'blue'  
#     elif row['price'] > 50:  # Expensive items in other categories
#         return 'green'  
#     else:
#         return 'gray'   

# # Generate colors for each row
# colors = [get_color(row) for _, row in data.iterrows()]

# # Set up the plot
# fig, ax = plt.subplots(figsize=(8, 5))

# # Plot parallel coordinates with custom colors
# for i, row in data_scaled.iterrows():
#     ax.plot(data_scaled.columns, row, color=colors[i], alpha=0.7)

# # Add min, 1/3, 2/3, and max ticks for numerical parameters
# for i, column in enumerate(['price', 'quality', 'category']):  
#     if column == "category":
#         # Get category labels and scale them evenly
#         category_labels = list(category_mapping.keys())
#         category_positions = np.linspace(0, 1, len(category_labels))  # Evenly spread positions

#         for pos, label in zip(category_positions, category_labels):
#             ax.text(i, pos, label, ha='center', va='center', fontsize=10, color='black')

#     else:
#         min_val, max_val = data[column].min(), data[column].max()
#         tick_values = np.linspace(min_val, max_val, 4)  # 4 evenly spread values
#         tick_positions = np.linspace(0, 1, 4)  # Normalized positions

#         for pos, val in zip(tick_positions, tick_values):
#             ax.text(i, pos, f"{val:.0f}", ha='center', va='center', fontsize=10, color='black')

#     ax.plot([i, i], [0, 1], color='black', linestyle='dashed', alpha=0.5)

# # Add x-axis labels
# ax.set_xticks(range(len(data_scaled.columns)))
# ax.set_xticklabels(data_scaled.columns, rotation=45)

# # Remove the black box (spines)
# for spine in ax.spines.values():
#     spine.set_visible(False)

# ax.set_ylabel("Normalized Scale (including Category)")
# ax.set_title("Parallel Coordinates Plot with Properly Aligned Categorical Axis")

# plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# # Load datasets
# chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv", delimiter=",", encoding="utf-8")
# chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding="utf-8")

# # Filter data for the specific Name
# filtered_experiments = chp_experiments[chp_experiments["Name"] == "Vartaverket KVV 8 "]
# filtered_outcomes = chp_outcomes[chp_experiments["Name"] == "Vartaverket KVV 8 "]

# # Ensure dataframes are aligned by index
# filtered_outcomes = filtered_outcomes.loc[filtered_experiments.index]

# # Select columns to plot
# columns_to_plot = ['capture_cost', 'penalty_services', 'penalty_biomass']

# # Normalize numerical values
# scaler = MinMaxScaler()
# data_scaled = pd.DataFrame(scaler.fit_transform(filtered_outcomes[columns_to_plot]), columns=columns_to_plot)

# # Define color function (adjust this condition as needed)
# def get_color(row):
#     # Check if category is a special case (if you want to do something with "category")
#     if "category" in row:
#         return 'green'  # Example color for category, modify as needed

#     # Example conditions to assign color
#     if row['capture_cost'] > 50 and row['penalty_services'] > 20:  
#         return 'red'  
#     elif row['capture_cost'] < 0:
#         return 'blue'
#     else:
#         return 'gray'

# # Generate colors based on conditions
# colors = [get_color(row) for _, row in filtered_outcomes.iterrows()]

# # Set up plot
# fig, ax = plt.subplots(figsize=(8, 5))

# # Plot parallel coordinates
# for i, row in data_scaled.iterrows():
#     ax.plot(data_scaled.columns, row, color=colors[i], alpha=0.7)

# # Add min, 1/3, 2/3, and max tick labels for each parameter
# for i, column in enumerate(columns_to_plot):
#     min_val, max_val = filtered_outcomes[column].min(), filtered_outcomes[column].max()
#     tick_values = np.linspace(min_val, max_val, 4)  # Evenly spaced values
#     tick_positions = np.linspace(0, 1, 4)  # Normalized positions

#     for pos, val in zip(tick_positions, tick_values):
#         ax.text(i, pos, f"{val:.1f}", ha='center', va='center', fontsize=10, color='black')

#     ax.plot([i, i], [0, 1], color='black', linestyle='dashed', alpha=0.5)

# # Add x-axis labels
# ax.set_xticks(range(len(data_scaled.columns)))
# ax.set_xticklabels(data_scaled.columns, rotation=45)

# # Remove black box (spines)
# for spine in ax.spines.values():
#     spine.set_visible(False)

# ax.set_ylabel("Normalized Scale")
# ax.set_title("Parallel Coordinates Plot for CHP Outcomes (Filtered by Vartaverket KVV 8)")

# plt.show()


# -------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# # Load datasets
# chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv", delimiter=",", encoding="utf-8")
# chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding="utf-8")

# # Filter data for the specific Name
# filtered_experiments = chp_experiments[chp_experiments["Name"] == "Vartaverket KVV 8 "]
# filtered_outcomes = chp_outcomes[chp_experiments["Name"] == "Vartaverket KVV 8 "]

# # Ensure dataframes are aligned by index
# filtered_outcomes = filtered_outcomes.loc[filtered_experiments.index]

# # Add the 'rate' column from experiments to outcomes
# filtered_outcomes['rate'] = filtered_experiments['rate']
# filtered_outcomes['duration_increase'] = filtered_experiments['duration_increase']
# filtered_outcomes['heat_pump'] = filtered_experiments['heat_pump']

# # Select columns to plot
# numerical_columns = ['capture_cost', 'penalty_services', 'penalty_biomass', 'rate']
# categorical_columns = ['duration_increase', 'heat_pump']

# # Normalize numerical values using MinMaxScaler
# scaler = MinMaxScaler()
# data_scaled_numerical = pd.DataFrame(scaler.fit_transform(filtered_outcomes[numerical_columns]), columns=numerical_columns)

# # Normalize categorical columns using LabelEncoder and then MinMaxScaler
# label_encoder = LabelEncoder()

# # Apply LabelEncoder to categorical columns and then normalize using MinMaxScaler
# categorical_data = filtered_experiments[categorical_columns].apply(label_encoder.fit_transform)
# categorical_scaled = pd.DataFrame(scaler.fit_transform(categorical_data), columns=categorical_columns)

# # Combine numerical and categorical data into one dataframe
# data_scaled = pd.concat([data_scaled_numerical, categorical_scaled], axis=1)
# print("Encoded values for 'duration_increase':", categorical_data['duration_increase'].unique())

# # Define color function (adjust this condition as needed)
# def get_color(row):
#     # # Example conditions to assign color
#     # if row['capture_cost'] > 120:  
#     #     return 'red'  
#     # elif row['capture_cost'] < 0:
#     #     return 'blue'
#     # else:
#     #     return 'gray'

#     if row['heat_pump'] == 1 and row['duration_increase'] == 2:  
#         return 'red'  
#     elif row['capture_cost'] < 0:
#         return 'blue'
#     else:
#         return 'gray'

# # Generate colors based on conditions
# colors = [get_color(row) for _, row in filtered_outcomes.iterrows()]

# # Set up plot
# fig, ax = plt.subplots(figsize=(8, 5))

# # Plot parallel coordinates
# for i, row in data_scaled.iterrows():
#     ax.plot(data_scaled.columns, row, color=colors[i], alpha=0.3)

# # Add min, 1/3, 2/3, and max tick labels for each parameter
# for i, column in enumerate(data_scaled.columns):
#     if column in numerical_columns:
#         min_val, max_val = filtered_outcomes[column].min(), filtered_outcomes[column].max()
#         tick_values = np.linspace(min_val, max_val, 4)  # Evenly spaced values
#         tick_positions = np.linspace(0, 1, 4)  # Normalized positions

#         for pos, val in zip(tick_positions, tick_values):
#             ax.text(i, pos, f"{val:.1f}", ha='center', va='center', fontsize=10, color='black')

#         ax.plot([i, i], [0, 1], color='black', linestyle='dashed', alpha=0.5)

# # Add x-axis labels
# ax.set_xticks(range(len(data_scaled.columns)))
# ax.set_xticklabels(data_scaled.columns, rotation=45)

# # Remove black box (spines)
# for spine in ax.spines.values():
#     spine.set_visible(False)

# ax.set_ylabel("Normalized Scale")
# ax.set_title("Parallel Coordinates Plot for CHP Outcomes (Filtered by Vartaverket KVV 8)")

# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.cm as cm  # Import colormap functions

# Load datasets
chp_experiments = pd.read_csv("PULP experiments/all_experiments.csv", delimiter=",", encoding="utf-8")
chp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding="utf-8")

# Filter data for the specific Name
filtered_experiments = chp_experiments[chp_experiments["Name"] == "Ostrand"]
filtered_outcomes = chp_outcomes[chp_experiments["Name"] == "Ostrand"]

# --- NEW: Randomly sample a hardcoded fraction of data ---
fraction_to_plot = 0.7  # Plot only 60% of rows
ostr_indices = chp_experiments[chp_experiments["Name"] == "Ostrand"].index
sampled_indices = np.random.choice(ostr_indices, size=int(len(ostr_indices) * fraction_to_plot), replace=False)
filtered_experiments = chp_experiments.loc[sampled_indices]
filtered_outcomes = chp_outcomes.loc[sampled_indices]

# Ensure dataframes are aligned by index
filtered_outcomes = filtered_outcomes.loc[filtered_experiments.index]

# Add the 'rate' column from experiments to outcomes
filtered_outcomes['rate'] = filtered_experiments['rate']
filtered_outcomes['beta'] = filtered_experiments['beta']
# filtered_outcomes['duration_increase'] = filtered_experiments['duration_increase']
filtered_outcomes['SupplyStrategy'] = filtered_experiments['SupplyStrategy']
filtered_outcomes['BarkIncrease'] = filtered_experiments['BarkIncrease']
filtered_outcomes['celc'] = filtered_experiments['celc']

# Select columns to plot
numerical_columns = ['capture_cost', 'penalty_services', 'penalty_biomass']
# numerical_columns = ['capture_cost', 'penalty_services', 'penalty_biomass', 'time', 'celc']
# categorical_columns = ['duration_increase', 'heat_pump']
categorical_columns = ['SupplyStrategy']

# Normalize numerical values using MinMaxScaler
scaler = MinMaxScaler()
data_scaled_numerical = pd.DataFrame(scaler.fit_transform(filtered_outcomes[numerical_columns]), columns=numerical_columns)
viridis = cm.get_cmap('cividis')

# capture_costs = filtered_outcomes['capture_cost']
# capture_costs_norm = (capture_costs - capture_costs.min()) / (capture_costs.max() - capture_costs.min())  # Normalize 0-1
# colors = cm.viridis(capture_costs_norm)  # Use colormap (viridis) to assign colors

# Normalize categorical columns using LabelEncoder and then MinMaxScaler
label_encoder = LabelEncoder()

# Apply LabelEncoder to categorical columns and then normalize using MinMaxScaler
categorical_data = filtered_experiments[categorical_columns].apply(label_encoder.fit_transform)
categorical_scaled = pd.DataFrame(scaler.fit_transform(categorical_data), columns=categorical_columns)

# Combine numerical and categorical data into one dataframe
data_scaled = pd.concat([data_scaled_numerical, categorical_scaled], axis=1)

def get_color(row):
    # General results
    # if row['SupplyStrategy'] == "SteamHP" and (row['BarkIncrease']==0 or row['BarkIncrease']==30):
    #     return "crimson", 1
    # elif row['SupplyStrategy'] == "SteamHP" and (row['BarkIncrease']==60 or row['BarkIncrease']==90):
    #     return "crimson", 0.05
    # elif row['SupplyStrategy'] == "SteamLP" and (row['BarkIncrease']==0 or row['BarkIncrease']==30):
    #     return "deepskyblue", 1
    # elif row['SupplyStrategy'] == "SteamLP" and (row['BarkIncrease']==60 or row['BarkIncrease']==90):
    #     return "deepskyblue", 0.05
    # elif row['SupplyStrategy'] == "HeatPumps" and (row['BarkIncrease']==0 or row['BarkIncrease']==30):
    #     return "mediumseagreen", 1
    # elif row['SupplyStrategy'] == "HeatPumps" and (row['BarkIncrease']==60 or row['BarkIncrease']==90):
    #     return "mediumseagreen", 0.05 

    # SD results
    if row['celc']<74 and row['SupplyStrategy']=="SteamLP" and (row['BarkIncrease']==0):
        return "deepskyblue", 1
    else:
        return "grey", 0.05

# Generate colors based on conditions
colors = [get_color(row) for _, row in filtered_outcomes.iterrows()]

# Set up plot (Assuming you want to plot, like in previous examples)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

# Plot parallel coordinates (assuming you already have data_scaled and colors from earlier)
for i, row in data_scaled.iterrows():
    ax.plot(data_scaled.columns, row, color=colors[i][0], alpha=colors[i][1])

# Add min, 1/3, 2/3, and max tick labels for each parameter (as in previous examples)
for i, column in enumerate(data_scaled.columns):
    if column in numerical_columns:
        min_val, max_val = filtered_outcomes[column].min(), filtered_outcomes[column].max()
        tick_values = np.linspace(min_val, max_val, 4)  # Evenly spaced values
        tick_positions = np.linspace(0, 1, 4)  # Normalized positions

        for pos, val in zip(tick_positions, tick_values):
            ax.text(i-0.1, pos, f"{int(round(val))}", ha='center', va='center', fontsize=10, color='black')
            ax.plot([i - 0.025, i + 0.025], [pos, pos], color='black', linewidth=1)  # Small horizontal tick
    ax.plot([i, i], [0, 1], color='black', linestyle='-', alpha=1)

# Add x-axis labels
ax.set_xticks(range(len(data_scaled.columns)))
ax.set_xticklabels(data_scaled.columns, rotation=45)

# Remove black box (spines)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_ylabel("Normalized Scale")
ax.set_title("Parallel Coordinates Plot for PULP Outcomes")
# Remove y-axis completely
ax.set_ylabel("")      # Remove the y-axis label
ax.set_yticks([])      # Remove y-axis tick marks
ax.spines['left'].set_visible(False)  # Hide the left spine (axis line)

plt.savefig("parallel_pulp.png", dpi=600, bbox_inches='tight')
plt.show()

