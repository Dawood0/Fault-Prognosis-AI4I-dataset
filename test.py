import pandas as pd
from scipy import stats
from sklearn.utils import resample
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def viewing_outlier(data):
    import os
    # data = pd.read_csv("Processed_data.csv")
    os.makedirs("outlier_figs", exist_ok=True)

    # Plotting and saving box plot for each column
    for column in data.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column])
        plt.title(f'Box Plot for {column}')
        plt.xlabel(column)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()
        # plt.savefig(f"outlier_figs/{column}_boxplot.png")
        plt.close()
        plt.show()

    # # Plotting and saving violin plot for each column
    # for column in data.columns:
    #     plt.figure(figsize=(8, 6))
    #     sns.violinplot(x=data[column])
    #     plt.title(f'Violin Plot for {column}')
    #     plt.xlabel(column)
    #     plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    #     plt.tight_layout()
    #     # plt.savefig(f"outlier_figs/{column}_violinplot.png")
    #     plt.close() 
    #     plt.show()
    print("Plots saved successfully!")

df = pd.read_csv('ai4i2020 - backup.csv')
selected_columns = [ 'Process temperature [K]']

# Extract the selected columns
df = df[selected_columns]
# viewing_outlier(df)


import pandas as pd
from scipy.stats import entropy
import numpy as np

# Load the dataset from the CSV file
data = pd.read_csv("ai4i2020 - backup.csv")

# Assuming the column containing the labels/targets is named 'label', replace it with the actual column name if different
labels = data['Machine failure']

# Calculate the entropy
entropy_value = entropy(np.bincount(labels), base=2)

print("Entropy:", entropy_value)