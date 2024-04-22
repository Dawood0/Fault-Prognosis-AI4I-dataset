import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the data from the CSV file
data = pd.read_csv("ai4i2020.csv")

# Select relevant columns for modeling
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression classifier
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Predictions on the testing set
y_pred = log_reg.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
