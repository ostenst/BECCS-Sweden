import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import random

def filter_dataframes(experiments_df, outcomes_df, conditions, categorical_conditions=None):
    """
    Filters the experiments and outcomes dataframes based on the specified conditions applied to the experiments dataframe.

    Parameters:
    experiments_df (pd.DataFrame): The experiments dataframe to be filtered.
    outcomes_df (pd.DataFrame): The outcomes dataframe to be filtered.
    conditions (dict): A dictionary where keys are column names and values are tuples of the form (min_value, max_value).
                       Use None for no limit on min_value or max_value.
    categorical_conditions (dict): A dictionary where keys are column names and values are lists of allowed categories.

    Returns:
    pd.DataFrame, pd.DataFrame: Two dataframes filtered based on the conditions - the experiments and outcomes dataframes.
    """
    # Build the query string for numerical conditions
    query = []
    for column, (min_value, max_value) in conditions.items():
        if min_value is not None:
            query.append(f"{column} >= {min_value}")
        if max_value is not None:
            query.append(f"{column} <= {max_value}")
    
    # Apply numerical conditions
    if query:
        query_string = " & ".join(query)
        filtered_experiments_df = experiments_df.query(query_string)
    else:
        filtered_experiments_df = experiments_df

    # Apply categorical conditions
    if categorical_conditions:
        for column, allowed_values in categorical_conditions.items():
            filtered_experiments_df = filtered_experiments_df[filtered_experiments_df[column].isin(allowed_values)]

    # Filter outcomes dataframe based on the filtered experiments dataframe
    filtered_outcomes_df = outcomes_df.loc[filtered_experiments_df.index]
    
    if filtered_experiments_df.empty:
        print("Filtered dataframe is empty")
        
    return filtered_experiments_df, filtered_outcomes_df

def plot_minmax_values(outcomes_df, KPI):
    # Group by 'Name' and calculate the min, max, and mean of capture_cost or penalty_services/biomass
    grouped = outcomes_df.groupby('Name')[KPI].agg(['min', 'max', 'mean'])
    grouped_nominal = outcomes_df.groupby('Name')['nominal'].mean()
    grouped = pd.concat([grouped, grouped_nominal], axis=1)
    grouped = grouped.sort_values(by='nominal', ascending=False)

    if KPI == "capture_cost":
        unit = "[EUR/t]"
    else:
        unit = "[MWh/kt]"

    plt.figure(figsize=(10, 6))  
    plt.scatter(grouped.index, grouped['min'], marker='o', color='b', label='Min')
    plt.scatter(grouped.index, grouped['max'], marker='s', color='g', label='Max')
    plt.scatter(grouped.index, grouped['mean'], marker='^', color='r', label='Mean')

    plt.xlabel('Name')
    plt.ylabel(f'{KPI} {unit}')
    plt.title(f'Min, Max, and Mean Values of {KPI}')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.legend()
    plt.tight_layout()
    plt.grid(True) 

def plot_satisficing(outcomes_df, thresholds):
    # Calculate total counts per 'Name' before masking
    total = outcomes_df.groupby('Name').size()
    total = total.reset_index(name='Total')

    mask = (outcomes_df['capture_cost'] < thresholds['capture_cost']) & \
           (outcomes_df['penalty_services'] < thresholds['penalty_services']) & \
           (outcomes_df['penalty_biomass'] < thresholds['penalty_biomass'])
    filtered_df = outcomes_df[mask]

    # Apply the mask and group by 'Name' to count rows meeting all conditions
    grouped_nominal_mean = outcomes_df.groupby('Name')['gross'].mean() #NOTE: this should maybe be calculated on the outcomes_df instead? Before masking.
    grouped = filtered_df.groupby('Name').size()
    grouped_df = grouped.reset_index(name='Satisficing')

    for name in total['Name']: # Ensures that all plants in total are put in the grouped_df, but with 0 satisficing scenarios
        if name not in grouped_df['Name'].values:
            new_row = pd.DataFrame({'Name': [name], 'Satisficing': [0]})
            grouped_df = pd.concat([grouped_df, new_row], ignore_index=True)

    satisficing_df = pd.merge(grouped_df, grouped_nominal_mean, on='Name', suffixes=('', '_mean'))
    satisficing_df.columns = ['Name', 'Satisficing', 'Gross CO2']

    satisficing_df = pd.merge(satisficing_df, total, on='Name')
    satisficing_df = satisficing_df.sort_values(by='Gross CO2', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(satisficing_df['Name'], satisficing_df['Satisficing'], color='b')

    plt.xlabel('Name')
    plt.ylabel(f'N satisficing scenarios (N total varies after masking)')
    plt.title('Satisficing scenarios per plant (sorted by gross CO2 [kt/yr])')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()

    if satisficing_df.empty:
        print("Satisficing dataframe is empty")

    return satisficing_df

def plot_densitymap(satisficing_df, coordinates_df):
    # Plotting maps (https://simplemaps.com/gis/country/se#admin1)
    sweden = gpd.read_file('se_shp', layer='se', crs='epsg:4326')

    coordinates_df = pd.merge(coordinates_df, satisficing_df, on='Name', how='left')
    coordinates_df['Density'] = coordinates_df['Satisficing'] / coordinates_df['Total']

    geometry = [Point(xy) for xy in zip(coordinates_df['Longitude'], coordinates_df['Latitude'])]
    crs = 'epsg:4326' 
    coordinates_gdf = gpd.GeoDataFrame(coordinates_df, geometry=geometry, crs=crs)

    # Plotting the map of Sweden
    fig, ax = plt.subplots(figsize=(10, 8))
    sweden.plot(ax=ax, edgecolor='black', color='white')

    # Plot circles with varying diameter
    # norm = mcolors.Normalize(vmin=coordinates_gdf['Density'].min(), vmax=coordinates_gdf['Density'].max())
    cmap = plt.colormaps.get_cmap('coolwarm_r')
    cmap = plt.colormaps.get_cmap('RdYlGn')

    for idx, row in coordinates_gdf.iterrows():
        radius_x = row['Gross CO2'] / 750
        radius_y = radius_x / 2.1     # Hard coding this seems to work...

        color = cmap(row['Density'])
        edgecolor = tuple(max(0, c - 0.2) for c in color[:3]) + (color[3],)
        color = (*color[:3], 0.80) # Sets an alpha value
        edgecolor = color

        if "City" in row.index:
            Title = row["City"]
        else:
            Title = row["Name"]

        ellipse = Ellipse((row['Longitude'], row['Latitude']), width=radius_x * 2, height=radius_y * 2, edgecolor=edgecolor, facecolor=color, fill=True, linewidth=1.0)
        ax.add_patch(ellipse)

        if row['Gross CO2'] > 200: #Only [%] above 
            ax.text(row['Longitude'], row['Latitude'], f"{round(row['Density']*100)}%", fontsize=7, ha='center', va='center')
        # if row['Gross CO2'] > 330: #Only [kt] above
        #     ax.text(row['Longitude']+radius_x, row['Latitude'], f"{round(row['Gross CO2'])} kt", fontsize=7, ha='left', va='center')
        # if row['Gross CO2'] > 450: #Only name above
        #     # move this text 
        #     ax.text(row['Longitude']+radius_x, row['Latitude']+0.2, Title, fontsize=7, ha='left', va='center')
            
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)

    tick_labels = [f"{int(t * 100)}%" for t in cbar.get_ticks()]
    cbar.set_ticks(cbar.get_ticks())
    cbar.set_ticklabels(tick_labels)
    cbar.set_label(f"Density of satisficing scenarios out of ~{round( sum(satisficing_df['Total'])/len(satisficing_df) )} per plant")

    ax.set_title('Satisficing scenarios and annual CO2 emissions')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    fig.savefig('UNNAMED_map.png', dpi=600)

def plot_everything(experiments, outcomes, coordinates, numerical_restrictions, categorical_restrictions, satisficing_thresholds):
        
    experiments, outcomes = filter_dataframes(experiments, outcomes, numerical_restrictions, categorical_restrictions)
    print(len(experiments), "scenarios remain after restricting dimensions")

    plot_minmax_values(outcomes, "capture_cost")
    plot_minmax_values(outcomes, "penalty_services")
    plot_minmax_values(outcomes, "penalty_biomass")

    satisficing_df = plot_satisficing(outcomes, satisficing_thresholds)

    # Plot densities on Swedish map
    plot_densitymap(satisficing_df, coordinates)

# Read data
chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')
chp_coordinates = pd.read_csv('chp_coordinates.csv')

w2e_experiments = pd.read_csv("WASTE experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
w2e_outcomes = pd.read_csv("WASTE experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')
w2e_coordinates = pd.read_csv('w2e_coordinates.csv')

pulp_experiments = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')
pulp_coordinates = pd.read_csv('pulp_coordinates.csv')

numerical_restrictions = {
    # 'BarkIncrease': (None, 31),
    # 'celc': (20, 92),
    # 'cheat': (42, 150),
    # 'COP': (2.9, 3.80),
    # 'Tsupp': (83, 100),
    # 'rate': (0.78, 0.893),
    # 'i': (0.05, 0.10),
    # 'time': (4688, 5999),
    # "duration_increase": (None, 1001)
}
categorical_restrictions = {
    # "SupplyStrategy": ["SteamLP"],
    # "BarkIncrease": [0]
    # "heat_pump": [True],
    # "duration_increase": [1000]
}
# satisficing_thresholds_1 = {
#     'capture_cost': 100,
#     'penalty_services': 300,
#     'penalty_biomass': 500
# }
# satisficing_thresholds_2 = {
#     'capture_cost': 100,
#     'penalty_services': 300,
#     'penalty_biomass': 1
# }
satisficing_thresholds_3 = {
    'capture_cost': 80,
    'penalty_services': 450,
    'penalty_biomass': 200
}

# BELOW IS MADNESS; BUT IT WORKED

#Figure out what plants have above 300 kt/yr
grouped = chp_outcomes.groupby('Name')
gross_means = grouped['gross'].mean().sort_values()

high_gross_names = gross_means[(gross_means > 350)].index.tolist()
boolean = chp_outcomes['Name'].isin(high_gross_names)
filtered_outcomes_high = chp_outcomes[boolean].reset_index(drop=True)
filtered_experiments_high = chp_experiments[boolean].reset_index(drop=True)

mid_gross_names = gross_means[(gross_means > 150) & (gross_means < 350)].index.tolist()
boolean = chp_outcomes['Name'].isin(mid_gross_names)
filtered_outcomes_mid = chp_outcomes[boolean].reset_index(drop=True)
filtered_experiments_mid = chp_experiments[boolean].reset_index(drop=True)

low_gross_names = gross_means[(gross_means < 150)].index.tolist()
boolean = chp_outcomes['Name'].isin(low_gross_names)
filtered_outcomes_low = chp_outcomes[boolean].reset_index(drop=True)
filtered_experiments_low = chp_experiments[boolean].reset_index(drop=True)

# # plot_everything(filtered_experiments, filtered_outcomes, chp_coordinates, numerical_restrictions, categorical_restrictions, satisficing_thresholds_1)
# plot_everything(chp_experiments, chp_outcomes, chp_coordinates, numerical_restrictions, categorical_restrictions, satisficing_thresholds_1)
# plot_everything(w2e_experiments, w2e_outcomes, w2e_coordinates, numerical_restrictions, categorical_restrictions, satisficing_thresholds_2)
# plot_everything(pulp_experiments, pulp_outcomes, pulp_coordinates, numerical_restrictions, categorical_restrictions, satisficing_thresholds_3)

plot_minmax_values(pulp_outcomes, "capture_cost")
plot_minmax_values(pulp_outcomes, "penalty_services")
plot_minmax_values(pulp_outcomes, "penalty_biomass")


# # BELOW IS FOR REGULAR SCENARIO
# numerical_restrictions_1 = {
#     # 'COP': (3.28, 3.80),
#     'celc': (20, 73),
#     'rate': (0.79, 0.90),
# }
# categorical_restrictions_1 = {
#     # "SupplyStrategy": ["SteamLP"],
#     "BarkIncrease": [30]
# }

# numerical_restrictions_2 = {
#     'beta': (0.6, 0.675),
#     'celc': (20, 58),
#     'BarkIncrease': (None, 31),
# }
# categorical_restrictions_2 = {
#     # "SupplyStrategy": ["SteamLP"],
#     # "BarkIncrease": [0]
# }

# numerical_restrictions_3 = {
#     # 'rate': (0.82, 0.90),
#     'celc': (20, 63),
#     'BarkIncrease': (None, 31),
# }
# categorical_restrictions_3 = {
#     "SupplyStrategy": ["SteamLP"],
#     # "BarkIncrease": [0]
# }

# numerical_restrictions_4 = {
#     'i': (0.05, 0.095),
#     'celc': (20, 85),
#     # 'BarkIncrease': (None, 31),
# }
# categorical_restrictions_4 = {
#     "SupplyStrategy": ["HeatPumps"],
#     # "BarkIncrease": [0]
# }

# numerical_restrictions_5 = {
#     # 'COP': (2.9, 3.80),
#     # 'rate': (0.80, 94),
#     'celc': (20, 68),
#     'BarkIncrease': (29, 61),
# }
# categorical_restrictions_5 = {
#     "SupplyStrategy": ["SteamLP"],
#     # "BarkIncrease": [0]
# }

# numerical_restrictions_6 = {
#     # 'COP': (3.28, 3.80),
#     'celc': (20, 73),
#     'BarkIncrease': (None, 31),
# }
# categorical_restrictions_6 = {
#     "SupplyStrategy": ["SteamLP"],
#     # "BarkIncrease": [0]
# }

# numerical_restrictions_7 = {
#     'beta': (0.6, 0.66),
#     'celc': (20, 55),
#     'BarkIncrease': (None, 31),
# }
# categorical_restrictions_7 = {
#     # "SupplyStrategy": ["SteamLP"],
#     # "BarkIncrease": [0]
# }

# BELOW ARE ZERO BIOMASS SCENARIOS
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
separate_mills = []
for idx, Name in enumerate(pulp_coordinates["Name"]):
    pulp_experiments_subset = pulp_experiments[pulp_experiments["Name"] == Name].reset_index(drop=True)
    # print(pulp_experiments_subset) # This should print correctly for all iterations
    pulp_outcomes_subset = pulp_outcomes[pulp_outcomes["Name"] == Name].reset_index(drop=True)
    # print(restrictions[idx][0])
    separate_mills.append([pulp_experiments_subset, pulp_outcomes_subset, restrictions[idx][0], restrictions[idx][1]])

# Now process each subset
satisficing_combined = pd.DataFrame()
# Copy paste for all 3 subsets of CHPs:
for subset in separate_mills:
    print("Processing subset with numerical restrictions:", subset[2])
    print("Processing subset with categorical restrictions:", subset[3])
    experiments, outcomes = filter_dataframes(subset[0], subset[1], subset[2], subset[3])
    print(len(experiments), "scenarios remain after restricting dimensions")
    satisficing_df = plot_satisficing(outcomes, satisficing_thresholds_3)
    satisficing_combined = pd.concat([satisficing_combined, satisficing_df]).drop_duplicates(subset='Name')

plot_densitymap(satisficing_combined, pulp_coordinates)
plt.show()