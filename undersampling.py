import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle

# Read the data from the CSV file
data = pd.read_csv("pca_results.csv")

# Separate features (X) and target variable (y)
X = data.drop(columns=['class'])
y = data['class']

# Apply random undersampling
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Combine the resampled data into a DataFrame
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)

# Shuffle the rows randomly
resampled_data_shuffled = shuffle(resampled_data, random_state=42)

# Save the resampled and shuffled data to a new CSV file
resampled_data_shuffled.to_csv("pca_undersampled_shuffled_data.csv", index=False)
