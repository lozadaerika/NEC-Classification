import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the datasets
file_path_separable = 'A2-ring/A2-ring-separable.txt'
file_path_merged = 'A2-ring/A2-ring-merged.txt'
file_path_test = 'A2-ring/A2-ring-test.txt'

# Load data into DataFrames
df_separable = pd.read_csv(file_path_separable, delimiter='\t', header=None)
df_merged = pd.read_csv(file_path_merged, delimiter='\t', header=None)
df_test = pd.read_csv(file_path_test, delimiter='\t', header=None)

# Display the first few rows of each dataset
#print("Separable Dataset:")
#print(df_separable.head())

#print("\nMerged Dataset:")
#print(df_merged.head())

#print("\nTest Dataset:")
#print(df_test.head())

# Extract features and labels
X_train_separable = df_separable.iloc[:, :-1].values
y_train_separable = df_separable.iloc[:, -1].values

X_train_merged = df_merged.iloc[:, :-1].values
y_train_merged = df_merged.iloc[:, -1].values

X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

# Standardize features
scaler = StandardScaler()
X_train_separable = scaler.fit_transform(X_train_separable)
X_train_merged = scaler.fit_transform(X_train_merged)
X_test = scaler.transform(X_test)

# Train SVM classifiers
svm_classifier_separable = SVC(kernel='rbf', C=10,degree=3)
#svm_classifier_separable = SVC(kernel='rbf', C=1.0)
svm_classifier_merged = SVC(kernel='rbf', C=10,degree=3)
#svm_classifier_merged = SVC(kernel='rbf', C=1.0)

svm_classifier_separable.fit(X_train_separable, y_train_separable)
svm_classifier_merged.fit(X_train_merged, y_train_merged)

# Make predictions on the test set
y_pred_separable = svm_classifier_separable.predict(X_test)
y_pred_merged = svm_classifier_merged.predict(X_test)

# Evaluate the models
accuracy_separable = accuracy_score(y_test, y_pred_separable)
accuracy_merged = accuracy_score(y_test, y_pred_merged)

print(f"Accuracy on the test set for separable dataset: {accuracy_separable:.2f}")
print(f"Accuracy on the test set for merged dataset: {accuracy_merged:.2f}")

# Compute classification error
error_separable = 1 - accuracy_separable
error_merged = 1 - accuracy_merged

print(f"Classification error on the test set for separable dataset: {error_separable:.2%}")
print(f"Classification error on the test set for merged dataset: {error_merged:.2%}")

# Compute confusion matrix
conf_matrix_separable = confusion_matrix(y_test, y_pred_separable)
conf_matrix_merged = confusion_matrix(y_test, y_pred_merged)

print("\nConfusion Matrix for Separable Dataset:")
print(conf_matrix_separable)

print("\nConfusion Matrix for Merged Dataset:")
print(conf_matrix_merged)

# Print classification report for both datasets
print("\nClassification Report for Separable Dataset:")
print(classification_report(y_test, y_pred_separable))

print("\nClassification Report for Merged Dataset:")
print(classification_report(y_test, y_pred_merged))
