
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Read the data from the CSV file
# data = pd.read_csv("ai4i2020.csv")

# # Select relevant columns for modeling
# selected_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
#                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
#                     'Machine failure']

# # Extract the selected columns
# data_selected = data[selected_columns]

# # Drop any rows with missing values
# data_selected = data_selected.dropna()

# # Separate features (X) and target variable (y)
# X = data_selected.drop(columns=['Machine failure'])
# y = data_selected['Machine failure']

# # Apply one-hot encoding to the 'Type' column
# X_encoded = pd.get_dummies(X, columns=['Type'], drop_first=True)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2)

# # Initialize and train the Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=2)
# rf_classifier.fit(X_train, y_train)

# # Predictions on the testing set
# y_pred = rf_classifier.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))







# after pca
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Read the data from the CSV file
data = pd.read_csv("ai4i2020.csv")

# Select relevant columns for dimensionality reduction
selected_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
                    'Machine failure']

# Extract the selected columns
data_selected = data[selected_columns]

# Drop any rows with missing values
data_selected = data_selected.dropna()

# Separate features (X) and target variable (y)
X = data_selected.drop(columns=['Machine failure'])
y = data_selected['Machine failure']

# Apply one-hot encoding to the 'Type' column
column_trans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],  # The column numbers to be transformed (here is [0] but can be [0, 1, 2, ...])
    remainder='passthrough'  # Leave the rest of the columns untouched
)
X_encoded = column_trans.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Apply PCA
pca = PCA(n_components=2)  # Specify the number of components
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)

# Initialize and train the Random Forest classifier with PCA-transformed features
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_pca_train, y_train)

# Predictions on the testing set
y_pred = rf_classifier.predict(X_pca_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
