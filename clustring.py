import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read the data from the CSV file
# data = pd.read_csv("ai4i2020.csv")
data = pd.read_csv("undersampled_shuffled_data.csv")

# Select relevant columns for clustering
selected_columns = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Extract the selected columns
data_selected = data[selected_columns]

# Drop any rows with missing values
data_selected = data_selected.dropna()

# Scale the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Perform clustering with KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Visualize the clusters (2D plot)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Plot')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
