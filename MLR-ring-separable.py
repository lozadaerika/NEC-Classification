import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc,mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sys

from plots import Plots_Class  

plotsClass = Plots_Class()

# Load dataset
file_path = 'A2-ring/A2-ring-separable-normalized.csv'

fileName="images/MLR/"+file_path.split("/")[1].split(".")[0]

label="Ring_separable"

# Load data into DataFrames
df = pd.read_csv(file_path, delimiter=',')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(df.head())

#Separate train and test
X_train, X_test, y_train, y_test  = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

plotsClass.printPlots(df,label,fileName)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
plotsClass.plot_pca(X_pca,y_train,"",fileName)
# Linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Binarize predictions based on a threshold
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(int)
# Convert true labels to binary
y_test_binary = (y_test > threshold).astype(int)

# Cross-validation score 5-fold
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5) 
# Display the cross-validation scores
print("Cross-Validation Scores:",cross_val_scores)
mean_cv_score = np.mean(cross_val_scores)
print(f"Mean Cross-Validation Score: {mean_cv_score:.4f}")

plotsClass.printPredictionPlots(X_test,y_test_binary,y_pred_binary,label,fileName)

# Confusion matrix
conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
print(f"\nConfusion Matrix for Dataset:")
print(conf_matrix)
plotsClass.plot_conf_matrix(conf_matrix,fileName)

# Classification error
E = 100 * (conf_matrix[0][1] + conf_matrix[1][0]) / (conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[0][1] + conf_matrix[1][0])
print("Classification error: ", E)

# Classification report
print(f"\nClassification Report for Dataset:")
classif_report = classification_report(y_test_binary, y_pred_binary)
print(classif_report)

# Evaluate the model
mse = mean_squared_error(y_test_binary, y_pred_binary)
r2 = r2_score(y_test_binary, y_pred_binary)
print("Mean Squared Separable Error:", mse)
print("R-squared Separable:", r2)

# Print the coefficients and intercept
print("Coefficients Separable: ", model.coef_)
print("Intercept Separable:", model.intercept_)

# Accuracy score (discrete values)
accuracy = accuracy_score(y_test_binary, y_pred_binary)
print("Accuracy:", accuracy)

# ROC curve and AUC (discrete values)
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_binary)
roc_auc = auc(fpr, tpr)
print(f'ROC AUC: {roc_auc}')
plotsClass.plot_roc(fpr,tpr,roc_auc,fileName)


with open(fileName+"-output.txt", 'w') as file:      
    sys.stdout = file  # Redirect stdout
    print("Cross-Validation Scores:", cross_val_scores)
    print("Mean Cross-Validation Score:", mean_cv_score)
    print(f"\nConfusion Matrix for {label} Dataset:")
    print(conf_matrix)
    print("Classification error: ", E)
    print(f"\nClassification Report for {label} Dataset:")
    print(classif_report)
    print("Mean Squared Separable Error:", mse)
    print("R-squared Separable:", r2)
    print("Accuracy:", accuracy*100)
    print(f'ROC AUC: {roc_auc}')

sys.stdout = sys.__stdout__