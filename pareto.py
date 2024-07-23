import pandas as pd
# def binning_biomass(group):
#     # I think the bins are ok now???
#     zero_values = group[group['penalty_biomass'] == 0]
#     non_zero_values = group[group['penalty_biomass'] != 0]
    
#     if non_zero_values.empty:
#         # This is True for Aspa, and for W2E plants 
#         zero_values['biomass_bins'] = pd.Interval(left=-float('inf'), right=float('inf'), closed='right')
#         combined = zero_values
#     else:
#         # Define bin edges
#         bin_edges = pd.qcut(non_zero_values['penalty_biomass'], q=8, retbins=True)[1]
#         bin_edges = [-float('inf')] + list(bin_edges) + [float('inf')] # WHEN THIS CREATES 8-9 BINS???
#         # bin_edges = list(bin_edges) + [float('inf')] # WHY DOES THIS ONLY CREATE 2 BINS
        
#         # Print bin edges for debugging
#         print(f"Bin edges for group '{group['Name'].iloc[0]}': {bin_edges}")
#         print("The bins are now ok...")

#         # Assign bins to non-zero values
#         non_zero_values['biomass_bins'] = pd.cut(non_zero_values['penalty_biomass'], bins=bin_edges, include_lowest=True)
        
#         # Assign zero values to the first bin
#         if not zero_values.empty:
#             zero_values['biomass_bins'] = pd.Interval(left=-float('inf'), right=bin_edges[1], closed='right')
#             # Could this be adding a new bin?
        
#         # Combine the zero and non-zero value DataFrames
#         combined = pd.concat([zero_values, non_zero_values])

#     return combined

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
chp_outcomes = chp_outcomes.iloc[300:1000,:]
print(chp_outcomes)

def binning_biomass(group):
    # Define the number of bins you want for non-zero penalty_biomass values
    num_bins = 9
    
    # Ensure 'Name' column is preserved and accessible
    if 'Name' not in group.columns:
        raise ValueError("Column 'Name' not found in the group DataFrame.")
    
    # Print the minimum penalty_biomass for debugging
    min_penalty_biomass = group['penalty_biomass'].min()
    print(f"Min penalty_biomass for group '{group['Name'].iloc[0]}' is {min_penalty_biomass}")
    
    # Separate the group into zero and non-zero penalty_biomass
    zero_penalty = group[group['penalty_biomass'] == 0]
    non_zero_penalty = group[group['penalty_biomass'] > 0]
    
    # Create bins for non-zero penalty_biomass values
    if not non_zero_penalty.empty:
        bin_edges = pd.cut(non_zero_penalty['penalty_biomass'], bins=num_bins, retbins=True)[1]
        print(len(bin_edges))
        print(bin_edges) # THIS IS WHERE THE BINS ARE CREATED. ONE (OR MORE) ARE LOST SINCE NOT ALL BINS ARE FILLED WITH VALUES
        non_zero_penalty['biomass_bins'] = pd.cut(non_zero_penalty['penalty_biomass'], bins=bin_edges)
        
        # Create intervals from the bin edges
        bin_intervals = pd.IntervalIndex.from_breaks(bin_edges)
        bin_labels = bin_intervals.to_list()
        bin_index = {label: idx for idx, label in enumerate(bin_labels)}
        
        # print(f"\nBin intervals for group '{group['Name'].iloc[0]}': {bin_intervals}")
        print(f"\nBin labels and indices: {bin_index}")
    else:
        bin_intervals = pd.IntervalIndex([])
        bin_index = {}
        print(f"No non-zero penalty_biomass values for group '{group['Name'].iloc[0]}'")
    
    # Assign a separate bin label for zero penalty_biomass values
    zero_penalty['biomass_bins'] = 'Zero Penalty'
    
    # Concatenate the zero and non-zero groups back together
    binned_group = pd.concat([zero_penalty, non_zero_penalty])
    
    # Print the number of unique bins
    print(f"Total unique bins for group '{group['Name'].iloc[0]}': {len(binned_group['biomass_bins'].unique())}")
    
    return binned_group

# Apply the binning function to each group
chp_outcomes = chp_outcomes.groupby('Name').apply(binning_biomass)
chp_outcomes = chp_outcomes.reset_index(drop=True)
for name, group in chp_outcomes.groupby('Name'):
    print(f"\nBins for plant '{name}':")
    print(len(group['biomass_bins'].unique()))


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

print("\nTotal Mean Captured for each 'Name':")
print(total_captured)

# Define a function to calculate the midpoint of a bin range
def get_bin_midpoint(bin_interval):
    if bin_interval == 'Zero Penalty':
        return 0
    return (bin_interval.left + bin_interval.right) / 2

stats_df['Mean Biomass'] = stats_df['Biomass Bin'].apply(get_bin_midpoint)
print("\nStatistics with Mean Biomass:")
print(stats_df)

# Time to summarize bins. Do it by estimating the captured volumes first, to append correct plants to the list.
name_list = [] 
estimated_capture = 0
i = 0
sorted_total_captured = total_captured.sort_values(by='Total Captured', ascending=False)

while estimated_capture < 130 and i < len(sorted_total_captured):
    estimated_capture += sorted_total_captured["Total Captured"].iloc[i]
    name_list.append(sorted_total_captured["Name"].iloc[i])
    i += 1
print("\nName list sorted by Total Captured and ensuring estimated_capture >= YOUR GUESS:")
print(name_list)