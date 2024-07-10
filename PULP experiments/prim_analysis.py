import pandas as pd
import matplotlib.pyplot as plt  
from ema_workbench.analysis import prim
print(" I GIVE UP THIS IS TOO MESSY BUT IT SHOULD BE SO EASY TO JUST ACCESS THE PRIM BOXES AND SAVE THEM")


plants = pd.read_csv("Pulp data.csv", delimiter=";")
all_experiments = pd.read_csv("all_experiments.csv", delimiter=",")
all_outcomes = pd.read_csv("all_outcomes.csv", delimiter=",")

# Selecting only some plants
plant_selector = (all_experiments["Name"] == plants["Name"][0])
print(f"Selected plant: {plants['Name'][0]}")
print(f"Number of selected rows: {plant_selector.sum()}")

all_experiments = all_experiments.loc[plant_selector]
all_outcomes = all_outcomes.loc[plant_selector]
assert len(all_experiments) == len(all_outcomes)

# Launch PRIM
y = (all_outcomes["capture_cost"] < 80) & (all_outcomes["penalty_services"] < 550) & (all_outcomes["penalty_biomass"] < 250)
x = all_experiments.iloc[:, 0:23]
print("The number of interesting cases are:\n", y.value_counts())

prim_alg = prim.Prim(x, y, threshold=0.8, peel_alpha=0.1) # Threshold was 0.8 before (Kwakkel)
box1 = prim_alg.find_box()

box1.show_ppt()             # Lines tradeoff
box1.show_tradeoff()        # Pareto tradeoff 
box1.write_ppt_to_stdout()  # Prints trajectory/tradeoff, useful!
# print(box1.box_lims) # list

# Gather all boxes in one dataframe
num_iterations = len(box1.peeling_trajectory)
all_dataframe = pd.DataFrame()

for i in range(num_iterations):
    all_dataframe.loc[i, 'Name'] = plants["Name"][0]  # Assign a constant Name

    t = box1.inspect(i, style="data")  # inspect() result

    if isinstance(t, tuple) and len(t) == 2:
        overall_stats_series, boxlims_df = t
        
        # Save overall statistics to columns in all_dataframe
        for stat_name, stat_value in overall_stats_series.items():
            all_dataframe.loc[i, f'Overall_{stat_name}'] = stat_value

        # Save boxlims to columns in all_dataframe if boxlims_df is not empty
        if not boxlims_df.empty:
            for col in boxlims_df.columns:
                all_dataframe.loc[i, f'Boxlims_{col}'] = boxlims_df[col].iloc[0]  # Assuming you want the first row values
        else:
            print(f"No boxlims data found for index {i}")
    else:
        print(f"Unexpected type or structure returned by inspect() at index {i}: {type(t)}")

# Now all_dataframe should contain the extracted data
print(all_dataframe)

# box1.show_ppt()             # Lines tradeoff
# box1.show_tradeoff()        # Pareto tradeoff 
# box1.write_ppt_to_stdout()  # Prints trajectory/tradeoff, useful!

# box1.select(6)              # I think we move to box nr 6
# box1.inspect_tradeoff()     # Print tradeoff to terminal, not useful?
# box1.show_pairs_scatter()   # I think we now plot nr 6 limits
# prim_alg.show_boxes(6) 
# box1.inspect(6)

# plt.show()
