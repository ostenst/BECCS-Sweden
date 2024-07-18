import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def filter_dataframes(experiments_df, outcomes_df, conditions):
    """
    Filters the experiments and outcomes dataframes based on the specified conditions applied to the experiments dataframe.

    Parameters:
    experiments_df (pd.DataFrame): The experiments dataframe to be filtered.
    outcomes_df (pd.DataFrame): The outcomes dataframe to be filtered.
    conditions (dict): A dictionary where keys are column names and values are tuples of the form (min_value, max_value).
                       Use None for no limit on min_value or max_value.

    Returns:
    pd.DataFrame, pd.DataFrame: Two dataframes filtered based on the conditions - the experiments and outcomes dataframes.
    """
    query = []
    for column, (min_value, max_value) in conditions.items():
        if min_value is not None:
            query.append(f"{column} >= {min_value}")
        if max_value is not None:
            query.append(f"{column} <= {max_value}")
    
    if query:
        query_string = " & ".join(query)
        filtered_experiments_df = experiments_df.query(query_string)
        filtered_outcomes_df = outcomes_df.loc[filtered_experiments_df.index]
    else:
        filtered_experiments_df = experiments_df
        filtered_outcomes_df = outcomes_df
    
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
    grouped = filtered_df.groupby('Name').size()
    grouped_nominal_mean = filtered_df.groupby('Name')['gross'].mean()

    grouped_df = grouped.reset_index(name='Satisficing')
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

    for idx, row in coordinates_gdf.iterrows():
        radius_x = row['Gross CO2'] / 1100
        radius_y = radius_x / 2.1     # Hard coding this seems to work...

        color = cmap(row['Density'])
        edgecolor = tuple(max(0, c - 0.2) for c in color[:3]) + (color[3],)
        color = (*color[:3], 0.90) # Sets an alpha value
        edgecolor = color

        ellipse = Ellipse((row['Longitude'], row['Latitude']), width=radius_x * 2, height=radius_y * 2, edgecolor=edgecolor, facecolor=color, fill=True, linewidth=1.0)
        ax.add_patch(ellipse)
        ax.text(row['Longitude'], row['Latitude'], f"{round(row['Density']*100)}%", fontsize=7, ha='center', va='center')
        ax.text(row['Longitude']+radius_x, row['Latitude'], f"{round(row['Gross CO2'])} kt", fontsize=7, ha='left', va='center')
        ax.text(row['Longitude']+radius_x, row['Latitude']+0.2, row["Name"], fontsize=7, ha='left', va='center')
        
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

conditions = {
    # 'BarkIncrease': (32, None),
    # 'celc': (None, 60),
    # Add more conditions as needed
}
# chp_experiments, chp_outcomes = filter_dataframes(chp_experiments, chp_outcomes, conditions)
print(len(pulp_experiments), "scenarios are considered")
pulp_experiments, pulp_outcomes = filter_dataframes(pulp_experiments, pulp_outcomes, conditions)
print(len(pulp_experiments), "scenarios remain after filtering")

# Plot explored KPIs
plot_minmax_values(chp_outcomes, "capture_cost")

# Plot satisficing scenarios
thresholds = {
    'capture_cost': 100,
    'penalty_services': 500,
    'penalty_biomass': 500
}

satisficing_chp = plot_satisficing(chp_outcomes, thresholds)
satisficing_pulp = plot_satisficing(pulp_outcomes, thresholds)
print(satisficing_pulp)

satisficing_combined = pd.concat([satisficing_chp, satisficing_pulp]).drop_duplicates(subset='Name')
coordinates_df = pd.read_csv('pulp_coordinates.csv')

# Plot densities on Swedish map
plot_densitymap(satisficing_combined, coordinates_df)

plt.show()