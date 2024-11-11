import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read data (assuming files are available and correctly formatted), then define X (input features) and y (binary outcome)
chp_experiments = pd.read_csv("WASTE experiments/all_experiments.csv", delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("WASTE experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

pulp_experiments = pd.read_csv("PULP experiments/all_experiments.csv",delimiter=",", encoding='utf-8')
pulp_outcomes = pd.read_csv("PULP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

# # If you want a specific plant
# pulp_experiments = pulp_experiments[ (pulp_experiments["Name"] == "Morrum") ].reset_index(drop=True)
# pulp_outcomes = pulp_outcomes[ (pulp_outcomes["Name"] == "Morrum") ].reset_index(drop=True)

# #Figure out what plants have above 300 kt/yr
# grouped = chp_outcomes.groupby('Name')
# gross_means = grouped['gross'].mean().sort_values()
# # high_gross_names = gross_means[(gross_means > 350)].index.tolist()
# high_gross_names = gross_means[(gross_means < 150)].index.tolist()
# # high_gross_names = gross_means[(gross_means < 350) & (gross_means > 150)].index.tolist()
# boolean = chp_outcomes['Name'].isin(high_gross_names)
# chp_outcomes = chp_outcomes[boolean].reset_index(drop=True)
# chp_experiments = chp_experiments[boolean].reset_index(drop=True)

# # Plot the feature importance
# plt.figure(figsize=(10, 6))
# Iterate over different scenarios

x = chp_experiments.iloc[:, 0:24]  # Input features, 0:26 for CHP, 0:24 for WASTE, 0.23 for PULP
y = (chp_outcomes["capture_cost"] < 120) & (chp_outcomes["penalty_services"] < 450) & (chp_outcomes["penalty_biomass"] < 1)
print(f"{y.sum()} scenarios are satisficing out of {len(y)}")

X_encoded = pd.get_dummies(x, drop_first=False)  # One-hot encode categorical variables
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, criterion="entropy")
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate feature importance means and standard deviations across trees
feature_importances = np.array([tree.feature_importances_ for tree in rf_model.estimators_])
mean_importance = feature_importances.mean(axis=0)
std_importance = feature_importances.std(axis=0)
sorted_idx = mean_importance.argsort()

# Plotting feature importances with error bars
plt.figure(figsize=(10, 5))
plt.barh(range(len(sorted_idx)), mean_importance[sorted_idx], xerr=std_importance[sorted_idx],
         color="crimson", alpha=1.0, align='center', capsize=4, error_kw={'elinewidth':1, 'alpha':0.9})
plt.yticks(range(len(sorted_idx)), [X_train.columns[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance (with Std Dev)')
plt.grid(axis='x', color='lightgrey', linestyle='--', linewidth=0.7)
plt.show()