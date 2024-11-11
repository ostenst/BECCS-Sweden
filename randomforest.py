import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read data (assuming files are available and correctly formatted), then define X (input features) and y (binary outcome)
chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv", delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

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

#  If you want only zero biomass scenarios
zero_biomass_boolean = (chp_experiments["duration_increase"] == 0)
chp_experiments = chp_experiments[zero_biomass_boolean].reset_index(drop=True)
chp_outcomes = chp_outcomes[zero_biomass_boolean].reset_index(drop=True)

# zero_biomass_boolean = (pulp_experiments["BarkIncrease"] == 0)
# pulp_experiments = pulp_experiments[zero_biomass_boolean].reset_index(drop=True)
# pulp_outcomes = pulp_outcomes[zero_biomass_boolean].reset_index(drop=True)


# Read data and define input features (X) and binary outcome (y)
x = chp_experiments.iloc[:, 0:26]  # Your input features
y = (chp_outcomes["capture_cost"] < 120) & (chp_outcomes["penalty_services"] < 350) & (chp_outcomes["penalty_biomass"] < 500)
print(f"{y.sum()} scenarios are satisficing out of {len(y)}")

# One-Hot Encode categorical variables in X
print("Categorical variables are encoded to binary columns. I am not dropping the first category, as there is no 'natural' baseline category. Do check this assumption before analyzing!")
X_encoded = pd.get_dummies(x, drop_first=False)  # drop_first=True avoids multicollinearity

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Standardize the features (optional, but often helps with models like Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, criterion="entropy")
rf_model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test_scaled)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importance from the Random Forest model
feature_importance = rf_model.feature_importances_

# Sort the feature importance in descending order
sorted_idx = feature_importance.argsort()

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [X_train.columns[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()