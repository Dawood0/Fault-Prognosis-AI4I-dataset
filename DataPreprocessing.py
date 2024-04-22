import pandas as pd
from scipy import stats
from sklearn.utils import resample
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates()

def remove_useless_columns(df):
    """
    Remove columns that are not needed for analysis.
    """
    columns_to_remove = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    return df.drop(columns=columns_to_remove)

def convert_type_to_digit(df):
    """
    Convert the 'Type' column from letters to digit values.
    """
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    df['Type'] = df['Type'].map(type_mapping)
    return df

def remove_outliers(df, threshold=3):
    """
    Remove outliers from the numerical columns using Z-score method.
    """
    z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
    outlier_indices = (abs(z_scores) > threshold).any(axis=1)
    return df[~outlier_indices]

def balance_data(df, target_column):
    """
    Balance the data using SMOTE (Synthetic Minority Over-sampling Technique).
    
    Parameters:
    - df: DataFrame containing the dataset.
    - target_column: Name of the target column in the DataFrame.
    
    Returns:
    - balanced_df: DataFrame with the balanced data.
    """
    # Separate features (X) and target variable (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Apply SMOTE for oversampling
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Create a new DataFrame with the balanced data
    balanced_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[target_column])], axis=1)
    
    return balanced_df

def shuffle_data(df):
    """
    Shuffle the data points in the DataFrame.
    
    Parameters:
    - df: DataFrame containing the dataset.
    
    Returns:
    - shuffled_df: DataFrame with shuffled data points.
    """
    shuffled_df = shuffle(df, random_state=42)
    return shuffled_df

def apply_pca_and_save(X,y, n_components, output_file):
    """
    Apply Principal Component Analysis (PCA) to reduce the dimensionality of the dataset
    and save the transformed features to a CSV file.
    
    Parameters:
    - X: Features of the dataset.
    - n_components: Number of principal components to retain.
    - output_file: File path to save the transformed features.
    """

    # Creating PCA object
    pca = PCA(n_components=n_components)
    
    # Fitting PCA to the data and transforming the data
    X_pca = pca.fit_transform(X)
    
    # Creating a DataFrame with the transformed features
    pca_columns = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    
    df_pca['Machine failure'] = y
    # Saving the transformed features to a CSV file
    shuffle(df_pca, random_state=42).to_csv(output_file, index=False)
    
    
    # print("Transformed features saved to:", output_file)

def correlation_matrix_heatmap(data):
    """
    Calculate the correlation matrix and generate a heatmap for visualization.
    
    Parameters:
    - data: DataFrame containing the dataset.
    """
    # Calculate the correlation matrix
    corr_matrix = data.corr()
    
    # Generate a heatmap for visualization
    plt.figure(figsize=(20, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

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
        plt.savefig(f"outlier_figs/{column}_boxplot.png")
        plt.close()

    # Plotting and saving violin plot for each column
    for column in data.columns:
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=data[column])
        plt.title(f'Violin Plot for {column}')
        plt.xlabel(column)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.savefig(f"outlier_figs/{column}_violinplot.png")
        plt.close() 
    print("Plots saved successfully!")

def clean_and_preprocess_data(df, outlier_threshold=3):
    """
    Perform data cleaning and preprocessing steps.
    """
    import time
    start_time = time.time()

    # Remove duplicates
    df = remove_duplicates(df)
    
    # Remove useless columns
    df = remove_useless_columns(df)
    
    # Convert 'Type' column to digit values
    df = convert_type_to_digit(df)
    
    # # Remove outliers
    # df = remove_outliers(df, threshold=outlier_threshold)

    # Balance the data using SMOTE
    df = balance_data(df,'Machine failure')

    df= shuffle_data(df)

    # # # Viewing the outliers
    viewing_outlier(df)

    # # Apply PCA and save the transformed features
    apply_pca_and_save(df.drop(columns=['Machine failure']), df['Machine failure'], 3, 'pca_results.csv')

    # # # Generate a correlation matrix heatmap
    # correlation_matrix_heatmap(df)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    return df

# Read the original dataset
df = pd.read_csv('ai4i2020 - backup.csv')

# Clean and preprocess the data
processed_df = clean_and_preprocess_data(df)

# Save the processed data to a new CSV file
processed_df.to_csv('Processed_data.csv', index=False)

print("Data cleaning and preprocessing completed. Processed data saved to 'Processed_data.csv'.")
