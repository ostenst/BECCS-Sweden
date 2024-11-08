# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils import resample
# import shap

# # Read data (assuming files are available and correctly formatted), then define X (input features) and y (binary outcome)
# chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv", delimiter=",", encoding='utf-8')
# chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

# x = chp_experiments.iloc[:, 0:26] 
# y = (chp_outcomes["capture_cost"] < 120) & (chp_outcomes["penalty_services"] < 350) & (chp_outcomes["penalty_biomass"] < 500)
# print(f"{y.sum()} scenarios are satisficing out of {len(y)}")

# # Split the data into training and testing sets (80% train, 20% test), then scale the features
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # Perform undersampling on the majority class (False)
# X_train_undersampled, y_train_undersampled = resample(
#     X_train[y_train == False], 
#     y_train[y_train == False], 
#     replace=False, 
#     n_samples=y_train[y_train == True].sum(), 
#     random_state=42
# )

# # Combine the undersampled majority class (False) with the minority class (True)
# X_train_balanced = pd.concat([X_train_undersampled, X_train[y_train == True]])
# y_train_balanced = pd.concat([y_train_undersampled, y_train[y_train == True]])

# # Shuffle the training set to mix False and True class instances
# X_train_balanced, y_train_balanced = resample(X_train_balanced, y_train_balanced, random_state=42)

# # Scale the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_balanced)
# X_test_scaled = scaler.transform(X_test)

# # Fit the logistic regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_scaled, y_train_balanced)

# # Predict and evaluate the model
# y_pred = model.predict(X_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

# # Print model evaluation results
# print(f"Accuracy: {accuracy:.4f}")
# print(f"ROC AUC: {roc_auc:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Calculate SHAP values using the trained model and the scaled test data
# explainer = shap.LinearExplainer(model, X_train_scaled)
# shap_values = explainer.shap_values(X_test_scaled)

# # Convert shap_values to the correct shape if necessary
# # For binary classification, shap_values is typically a list of arrays
# if isinstance(shap_values, list) and len(shap_values) > 1:
#     shap_values = shap_values[1]  # Select SHAP values for the positive class

# # Plot SHAP summary
# shap.summary_plot(shap_values, X_test_scaled, feature_names=x.columns)

print(" I think the RandomForest approach is better, as the classification model is impressively accurate/robust!" )
print(" Probably I should run it SEPARATELY for large/medium/small plants - that is what I do with PRIM anyway")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read data (assuming files are available and correctly formatted), then define X (input features) and y (binary outcome)
chp_experiments = pd.read_csv("CHP experiments/all_experiments.csv", delimiter=",", encoding='utf-8')
chp_outcomes = pd.read_csv("CHP experiments/all_outcomes.csv", delimiter=",", encoding='utf-8')

#Figure out what plants have above 300 kt/yr
grouped = chp_outcomes.groupby('Name')
gross_means = grouped['gross'].mean().sort_values()
high_gross_names = gross_means[(gross_means > 300)].index.tolist()
# high_gross_names = gross_means[(gross_means < 200)].index.tolist()
# high_gross_names = gross_means[(gross_means < 250) & (gross_means > 125)].index.tolist()
boolean = chp_outcomes['Name'].isin(high_gross_names)
chp_outcomes = chp_outcomes[boolean].reset_index(drop=True)
chp_experiments = chp_experiments[boolean].reset_index(drop=True)

x = chp_experiments.iloc[:, 0:26]  # Your input features
y = (chp_outcomes["capture_cost"] < 120) & (chp_outcomes["penalty_services"] < 350) & (chp_outcomes["penalty_biomass"] < 500)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features (optional, but often helps with models like Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
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
