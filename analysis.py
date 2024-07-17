import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.patches import Ellipse


def plot_minmax_values(outcomes_df, KPI):
    # Group by 'Name' and calculate the min, max, and mean of the specified cost_type
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
    mask = (outcomes_df['capture_cost'] < thresholds['capture_cost']) & \
       (outcomes_df['penalty_services'] < thresholds['penalty_services']) & \
       (outcomes_df['penalty_biomass'] < thresholds['penalty_biomass'])

    # Apply the mask and group by 'Name' to count rows meeting all conditions
    grouped = outcomes_df[mask].groupby('Name').size()

    # Calculate mean of 'nominal' for each 'Name'
    grouped_nominal_mean = outcomes_df[mask].groupby('Name')['gross'].mean()
    grouped = pd.concat([grouped, grouped_nominal_mean], axis=1)
    grouped.columns = ['Satisficing', 'Gross CO2']
    satisficing_df = grouped.sort_values(by='Gross CO2', ascending=False)

    total = len(outcomes_df) / outcomes_df['Name'].nunique()
    satisficing_df['Total'] = total

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(satisficing_df.index, satisficing_df['Satisficing'], color='b')

    plt.xlabel('Name')
    plt.ylabel(f'N satisficing scenarios (total = {int(total)})')
    plt.title('Satisficing scenarios per plant (sorted by gross CO2 [kt/yr])')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()

    return satisficing_df


# Read data
experiments_df = pd.read_csv("CHP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

experiments_df = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

# Plot explored KPIs
plot_minmax_values(chp_outcomes, "capture_cost")
# plot_minmax_values(pulp_outcomes, "capture_cost")

# plot_minmax_values(chp_outcomes, "penalty_services")
# plot_minmax_values(pulp_outcomes, "penalty_services")

# plot_minmax_values(chp_outcomes, "penalty_biomass")
# plot_minmax_values(pulp_outcomes, "penalty_biomass")

# Plot satisficing scenarios
thresholds = {
    'capture_cost': 100,
    'penalty_services': 500,
    'penalty_biomass': 500
}

satisficing_chp = plot_satisficing(chp_outcomes, thresholds)
satisficing_pulp = plot_satisficing(pulp_outcomes, thresholds)
satisficing_chp.reset_index(inplace=True)
satisficing_pulp.reset_index(inplace=True)

# Plotting maps (https://simplemaps.com/gis/country/se#admin1)
sweden = gpd.read_file('se_shp', layer='se', crs='epsg:4326')
coordinates_df = pd.read_csv('coordinates.csv')

# Annoying rows to add Gross CO2 emissions to coordinate df
coordinates_df = pd.merge(coordinates_df, satisficing_chp[['Name', 'Gross CO2']], on='Name', how='left')
coordinates_df = pd.merge(coordinates_df, satisficing_pulp[['Name', 'Gross CO2']], on='Name', how='left', suffixes=('', '_pulp'))
coordinates_df['Gross CO2'].fillna(coordinates_df['Gross CO2_pulp'], inplace=True)
coordinates_df.drop(['Gross CO2_pulp'], axis=1, inplace=True)

print(coordinates_df)

geometry = [Point(xy) for xy in zip(coordinates_df['Longitude'], coordinates_df['Latitude'])]
crs = 'epsg:4326' 
coordinates_gdf = gpd.GeoDataFrame(coordinates_df, geometry=geometry, crs=crs)


# Plotting the map of Sweden
fig, ax = plt.subplots(figsize=(10, 8))
sweden.plot(ax=ax, edgecolor='black', color='white')

extent = sweden.total_bounds
aspect_ratio = (extent[2] - extent[0]) / (extent[3] - extent[1])
print(aspect_ratio)
# Plot circles with varying diameter
for idx, row in coordinates_gdf.iterrows():
    radius_x = row['Gross CO2'] / 1000
    radius_y = radius_x / 2     # Hard coding this seems to work...
    ellipse = Ellipse((row['Longitude'], row['Latitude']), width=radius_x * 2, height=radius_y * 2, edgecolor='red', facecolor='blue', fill=True)
    ax.add_patch(ellipse)
    ax.text(row['Longitude'], row['Latitude'], row['Name'], fontsize=10, ha='center', va='center')

ax.set_title('Map of Sweden with Circles')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.tight_layout()  # Adjust layout for better spacing

plt.show()




# # Launching PRIM algorithm
# # zero_biomass_boolean = (df_experiments["BarkIncrease"] == "0")
# y = df_outcomes["capture_cost"] < 10
# # y = df_outcomes["penalty_services"]<300 
# # y = df_outcomes["penalty_biomass"]==0 
# # y = (df_outcomes["capture_cost"] < 90) & (df_outcomes["penalty_services"] < 600) & (df_outcomes["penalty_biomass"] < 220)
# # y = (all_outcomes["capture_cost"] < 100) & (all_outcomes["penalty_services"] < 500) 

# x = all_experiments.iloc[:, 0:23]
# print("The number of interesting cases are:\n", y.value_counts())

# prim_alg = prim.Prim(x, y, threshold=0.6, peel_alpha=0.1) # Threshold was 0.8 before (Kwakkel) #NOTE: To avoid deprecated error, I replaced line 1506 in prim.py with: np.int(paste_value) => int(paste_value)
# box1 = prim_alg.find_box()

# # plt.clf()
# box1.show_ppt()             # Lines tradeoff
# box1.show_tradeoff()        # Pareto tradeoff 
# box1.write_ppt_to_stdout()  # Prints trajectory/tradeoff, useful!

# box1.select(14)
# # box1.inspect_tradeoff()     # Print tradeoff to terminal, not useful?
# prim_alg.show_boxes(14) 
# prim_alg.boxes_to_dataframe() # Save boxes here?
# box1.inspect(14)

# all_experiments.to_csv("all_experiments.csv", index=False)
# all_outcomes.to_csv("all_outcomes.csv", index=False)
# plt.show()