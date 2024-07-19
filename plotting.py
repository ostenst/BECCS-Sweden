import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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
        radius_x = row['Gross CO2'] / 1100
        radius_y = radius_x / 2.1     # Hard coding this seems to work...

        color = cmap(row['Density'])
        edgecolor = tuple(max(0, c - 0.2) for c in color[:3]) + (color[3],)
        color = (*color[:3], 0.90) # Sets an alpha value
        edgecolor = color

        if "City" in row.index:
            Title = row["City"]
        else:
            Title = row["Name"]

        ellipse = Ellipse((row['Longitude'], row['Latitude']), width=radius_x * 2, height=radius_y * 2, edgecolor=edgecolor, facecolor=color, fill=True, linewidth=1.0)
        ax.add_patch(ellipse)
        
        if row['Gross CO2'] > 300: #Only numbers above 300kt
            ax.text(row['Longitude'], row['Latitude'], f"{round(row['Density']*100)}%", fontsize=7, ha='center', va='center')
            ax.text(row['Longitude']+radius_x, row['Latitude'], f"{round(row['Gross CO2'])} kt", fontsize=7, ha='left', va='center')
        if row['Gross CO2'] > 400: #Only name above 400kt
            ax.text(row['Longitude']+radius_x, row['Latitude']+0.2, Title, fontsize=7, ha='left', va='center')
            
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)

    tick_labels = [f"{int(t * 100)}%" for t in cbar.get_ticks()]
    cbar.set_ticks(cbar.get_ticks())
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Density of satisficing scenarios')

    ax.set_title('Satisficing scenarios and annual CO2 emissions')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()

# Read data
chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

pulp_experiments = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

# Apply the filter
# boolean = (pulp_experiments["SupplyStrategy"] == "SteamLP") # NOTE: Using this temporarily to just select LP scenarios
# pulp_experiments = pulp_experiments[boolean].reset_index(drop=True)
# pulp_outcomes = pulp_outcomes[boolean].reset_index(drop=True)

# zero_biomass_boolean = (pulp_experiments["BarkIncrease"] == 0)
# pulp_experiments = pulp_experiments[zero_biomass_boolean].reset_index(drop=True)
# pulp_outcomes = pulp_outcomes[zero_biomass_boolean].reset_index(drop=True)

conditions = {
    # 'BarkIncrease': (None, 31),
    # 'celc': (20, 72),
    # 'cheat': (42, 150),
    'COP': (3.22, 3.79),
    # 'beta': (0.60, 0.69),
    'rate': (0.78, 0.893),
    # 'factor_recovery': (0.39, 0.40)
}
categorical_conditions = {
    # "SupplyStrategy": ["SteamLP"],
    # "BarkIncrease": [0]
    "heat_pump": [True],
    "duration_increase": [0]
}
# pulp_experiments, pulp_outcomes = filter_dataframes(pulp_experiments, pulp_outcomes, conditions, categorical_conditions)
chp_experiments, chp_outcomes = filter_dataframes(chp_experiments, chp_outcomes, conditions, categorical_conditions)
print(len(chp_experiments), "scenarios remain after filtering")

# Plot explored KPIs
plot_minmax_values(chp_outcomes, "capture_cost")
plot_minmax_values(chp_outcomes, "penalty_services")
plot_minmax_values(chp_outcomes, "penalty_biomass")

# Plot satisficing scenarios
thresholds = {
    'capture_cost': 100,
    'penalty_services': 400,
    'penalty_biomass': 600
}

satisficing_chp = plot_satisficing(chp_outcomes, thresholds)
satisficing_pulp = plot_satisficing(pulp_outcomes, thresholds)
print(satisficing_chp)

satisficing_combined = pd.concat([satisficing_chp, satisficing_pulp]).drop_duplicates(subset='Name')
coordinates_df = pd.read_csv('chp_coordinates.csv')

# Plot densities on Swedish map
plot_densitymap(satisficing_combined, coordinates_df)

plt.show()