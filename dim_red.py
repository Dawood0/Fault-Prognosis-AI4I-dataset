import pandas as pd
import matplotlib.pyplot as plt
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

# View the weights of the original features in the principal components
print("Weights of the original features in the principal components:")
print(pca.components_)

# Visualize the result
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Plot')
plt.colorbar(label='Machine failure')
plt.grid(True)
plt.show()

pca_results = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])

# Add the target variable (machine failure) to the DataFrame labeled as 'class' and place it before the principal components
pca_results.insert(0, 'class', y)
# Save the results to a CSV file
pca_results.to_csv("pca_results.csv", index=False)





# for 1 pca
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer

# # Read the data from the CSV file
# data = pd.read_csv("ai4i2020.csv")

# # Select relevant columns for dimensionality reduction
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
# column_trans = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(), [0])],  # The column numbers to be transformed (here is [0] but can be [0, 1, 2, ...])
#     remainder='passthrough'  # Leave the rest of the columns untouched
# )
# X_encoded = column_trans.fit_transform(X)

# # Scale the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_encoded)

# # Apply PCA with one component
# pca = PCA(n_components=1)  # Specify the number of components
# X_pca = pca.fit_transform(X_scaled)

# # Visualize the result
# plt.figure(figsize=(10, 6))
# plt.scatter(X_pca[:, 0], [0] * len(X_pca), c=y, cmap='viridis', alpha=0.5)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Fixed Value')
# plt.title('PCA Plot with One Principal Component')
# plt.colorbar(label='Machine failure')
# plt.grid(True)
# plt.show()







# # for 3 pca
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # Importing 3D plotting library
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer

# # Read the data from the CSV file
# data = pd.read_csv("ai4i2020.csv")

# # Select relevant columns for dimensionality reduction
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
# column_trans = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(), [0])],  # The column numbers to be transformed (here is [0] but can be [0, 1, 2, ...])
#     remainder='passthrough'  # Leave the rest of the columns untouched
# )
# X_encoded = column_trans.fit_transform(X)

# # Scale the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_encoded)

# # Apply PCA with three components
# pca = PCA(n_components=3)  # Specify the number of components
# X_pca = pca.fit_transform(X_scaled)

# # Visualize the result
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.5)
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# ax.set_title('PCA Plot with Three Principal Components')
# fig.colorbar(scatter, label='Machine failure')
# plt.show()
