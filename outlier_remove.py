import pandas as pd
from scipy import stats

# Assuming 'df' is your DataFrame containing the dataset
# Replace 'your_dataset.csv' with the actual file path if reading from a CSV file
df = pd.read_csv('ai4i2020.csv')

# Define a threshold for Z-score
threshold = 4

# Calculate Z-scores for each numerical column
z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))

# Find rows with outliers in any numerical column based on the threshold
outlier_rows = df[(abs(z_scores) > threshold).any(axis=1)]

# Remove rows with outliers
df_cleaned = df.drop(outlier_rows.index)

# Print the rows containing outliers
print("Rows with outliers:")
print(outlier_rows)

# Print the cleaned DataFrame
print("Cleaned DataFrame after removing outliers:")
print(df_cleaned)

# If you want to overwrite the original DataFrame with the cleaned one, you can use:
# df = df_cleaned

# If you want to write the cleaned DataFrame to a new CSV file:
# df_cleaned.to_csv('cleaned_dataset.csv', index=False)
