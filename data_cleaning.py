# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy import stats

# Load the dataset
# df = pd.read_csv('ai4i2020_k.csv')
df = pd.read_csv('ai4i2020 - backup.csv')

# Data Cleaning
# Assuming 'Nan' values are represented as np.nan in DataFrame
dropped_values = df[df.isna().any(axis=1)]
print("Dropped values with missing values:")
print(dropped_values)


print(len(df))
df = df.dropna()
print(len(df))
df_cleaned = df.drop_duplicates()
print(len(df_cleaned))




# # Data Preprocessing
# # Remove outliers
# z_scores = stats.zscore(df)
# abs_z_scores = np.abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# df = df[filtered_entries]

# # Remove highly correlated features
# corr_matrix = df.corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# df = df.drop(df[to_drop], axis=1)

# # Split the data into train and test sets
# X = df.drop('failure', axis=1)
# y = df['failure']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model Selection & Evaluation
# # Random Forest
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# rf_predictions = rf.predict(X_test)
# print("Random Forest Classifier report: \n\n", classification_report(y_test, rf_predictions))

# # Support Vector Machine
# svm = SVC()
# svm.fit(X_train, y_train)
# svm_predictions = svm.predict(X_test)
# print("Support Vector Machine report: \n\n", classification_report(y_test, svm_predictions))