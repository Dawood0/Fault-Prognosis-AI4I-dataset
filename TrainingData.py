from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib

def train_test_random_forest(X, y):
    """
    Split the data into training and testing sets, train a Random Forest classifier,
    evaluate its performance on the testing set, plot the ROC curve, and print the result.
    
    Parameters:
    - X: Features of the dataset.
    - y: Target variable of the dataset.
    
    Returns:
    - rf_classifier: Trained Random Forest classifier.
    """
    # Splitting the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating and training the Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42,n_estimators=5)
    rf_classifier.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred_rf = rf_classifier.predict(X_test)

    # Evaluating the model
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    
    # Calculating ROC curve and AUC
    y_prob_rf = rf_classifier.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    # Confusion matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    # Assuming rf_classifier is your trained Random Forest classifier
    feature_importance = rf_classifier.feature_importances_
    import numpy as np
    # Sort feature importance in descending order
    sorted_indices = np.argsort(feature_importance)[::-1]

    # Extract feature names
    feature_names = X.columns  # Assuming X is your DataFrame of features

    # Plot feature importance
    # Plot feature importance
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title('Feature Importance - Random Forest')
    plt.gca().invert_yaxis()  # Invert y-axis to display features from top to bottom
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

    model_filename = 'models/random_forest_model.pkl'
    joblib.dump(rf_classifier, model_filename)
    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Random Forest')
    plt.savefig('confmatrix/confusion_matrix_random_forest.png')  # Save the plot as an image
    # plt.show()

    # Plotting ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_rf)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend(loc='lower right')
    classifier_name = 'Random Forest'
    plt.savefig(f'roc_curve_{classifier_name}.png')  # Save the plot as an image
    # plt.show()

    print("Random Forest Performance:")
    print("Accuracy:", accuracy_rf)
    print("Precision:", precision_rf)
    print("Recall:", recall_rf)
    print("F1 Score:", f1_rf)
    print('auc:',auc_rf)
    
    
    return rf_classifier

def train_test_svm(X, y):
    """
    Split the data into training and testing sets, train a Support Vector Machine (SVM) classifier,
    evaluate its performance on the testing set, plot the ROC curve, and print the result.
    
    Parameters:
    - X: Features of the dataset.
    - y: Target variable of the dataset.
    
    Returns:
    - svm_classifier: Trained SVM classifier.
    """
    # Splitting the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating and training the SVM classifier
    svm_classifier = SVC(kernel='linear', probability=True, random_state=42)  # Using a linear kernel
    svm_classifier.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred_svm = svm_classifier.predict(X_test)

    # Evaluating the model
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm)
    recall_svm = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)
    
    # Calculating ROC curve and AUC
    y_prob_svm = svm_classifier.predict_proba(X_test)[:, 1]
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    auc_svm = roc_auc_score(y_test, y_prob_svm)

    cm_rf = confusion_matrix(y_test, y_pred_svm)

    import shap
    # Initialize an explainer object with the trained model and the training data
    model=svm_classifier
    explainer = shap.Explainer(model, X_train)

    # Calculate SHAP values for the test data
    shap_values = explainer.shap_values(X_test)

    # Visualize the SHAP values for a specific prediction (e.g., the first prediction)
    shap.summary_plot(shap_values, X_test, plot_type="bar")


    model_filename = 'models/svm_model.pkl'
    joblib.dump(svm_classifier, model_filename)
    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - SVM')
    plt.savefig('confmatrix/confusion_matrix_svm.png')  # Save the plot as an image
    # plt.show()

    # Plotting ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_svm)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - SVM')
    plt.legend(loc='lower right')
    classifier_name = 'Support Vector Machine'
    plt.savefig(f'roc_curve_{classifier_name}.png')  # Save the plot as an image
    # plt.show()

    print("Support Vector Machine (SVM) Performance:")
    print("Accuracy:", accuracy_svm)
    print("Precision:", precision_svm)
    print("Recall:", recall_svm)
    print("F1 Score:", f1_svm)
    print('auc:',auc_svm)

    return svm_classifier

def train_test_gradient_boosting(X, y):
    """
    Split the data into training and testing sets, train a Gradient Boosting classifier,
    evaluate its performance on the testing set, plot the ROC curve, and print the result.
    
    Parameters:
    - X: Features of the dataset.
    - y: Target variable of the dataset.
    
    Returns:
    - gb_classifier: Trained Gradient Boosting classifier.
    """
    # Splitting the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating and training the Gradient Boosting classifier
    gb_classifier = GradientBoostingClassifier(random_state=42)
    gb_classifier.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred_gb = gb_classifier.predict(X_test)

    # Evaluating the model
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    precision_gb = precision_score(y_test, y_pred_gb)
    recall_gb = recall_score(y_test, y_pred_gb)
    f1_gb = f1_score(y_test, y_pred_gb)
    
    # Calculating ROC curve and AUC
    y_prob_gb = gb_classifier.predict_proba(X_test)[:, 1]
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
    auc_gb = roc_auc_score(y_test, y_prob_gb)

    cm_rf = confusion_matrix(y_test, y_pred_gb)


    model_filename = 'models/gradient_boosting_model.pkl'
    joblib.dump(gb_classifier, model_filename)
    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Gradient Boosting')
    plt.savefig('confmatrix/confusion_matrix_gradient_boosting.png')  # Save the plot as an image
    # plt.show()

    # Plotting ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_gb, tpr_gb, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_gb)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Gradient Boosting')
    plt.legend(loc='lower right')
    classifier_name = 'Gradient Boosting'
    plt.savefig(f'roc_curve_{classifier_name}.png')  # Save the plot as an image
    # plt.show()

    print("Gradient Boosting Performance:")
    print("Accuracy:", accuracy_gb)
    print("Precision:", precision_gb)
    print("Recall:", recall_gb)
    print("F1 Score:", f1_gb)
    print('auc:',auc_gb)

    return gb_classifier

def train_test_logistic_regression(X, y):
    """
    Split the data into training and testing sets, train a Logistic Regression classifier,
    evaluate its performance on the testing set, plot the ROC curve, and print the result.
    
    Parameters:
    - X: Features of the dataset.
    - y: Target variable of the dataset.
    
    Returns:
    - lr_classifier: Trained Logistic Regression classifier.
    """
    # Splitting the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating and training the Logistic Regression classifier
    lr_classifier = LogisticRegression(random_state=42,max_iter=1000)
    lr_classifier.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred_lr = lr_classifier.predict(X_test)
    
    # Evaluating the model
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    precision_lr = precision_score(y_test, y_pred_lr)
    recall_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    
    # Calculating ROC curve and AUC
    y_prob_lr = lr_classifier.predict_proba(X_test)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)

    cm_rf = confusion_matrix(y_test, y_pred_lr)

    import shap
    # Initialize an explainer object with the trained model and the training data
    model=lr_classifier
    explainer = shap.Explainer(model, X_train)

    # Calculate SHAP values for the test data
    shap_values = explainer.shap_values(X_test)

    # Visualize the SHAP values for a specific prediction (e.g., the first prediction)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    model_filename = 'models/logistic_regression_model.pkl'
    joblib.dump(lr_classifier, model_filename)
    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - logistic Regression')
    plt.savefig('confmatrix/confusion_matrix_logistic_regressioin.png')  # Save the plot as an image
    # plt.show()

    # Plotting ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_lr)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc='lower right')
    classifier_name = 'Logistic Regression'
    plt.savefig(f'roc_curve_{classifier_name}.png')  # Save the plot as an image
    # plt.show()

    print("Logistic Regression Performance:")
    print("Accuracy:", accuracy_lr)
    print("Precision:", precision_lr)
    print("Recall:", recall_lr)
    print("F1 Score:", f1_lr)
    print('auc:',auc_lr)
        

    return lr_classifier




def Train():
    # Read the data from the CSV file
    data = pd.read_csv("Processed_data.csv")

    # Select relevant columns for modeling
    selected_columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
                        'Machine failure']

    # Extract the selected columns
    data_selected = data[selected_columns]

    X = data_selected.drop(columns=['Machine failure'])
    y = data_selected['Machine failure']

    # Assuming 'X' contains features and 'y' contains the target variable of the dataset
    # Replace 'X' and 'y' with your actual features and target variable
    
    start_time = time.time()
    trained_rf = train_test_random_forest(X, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    print('-------------------------------')

    # start_time = time.time()
    # trained_gb = train_test_gradient_boosting(X, y)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("Elapsed time:", elapsed_time, "seconds")
    # print('-------------------------------')

    # start_time = time.time()
    # trained_lr = train_test_logistic_regression(X, y)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("Elapsed time:", elapsed_time, "seconds")
    # print('-------------------------------')

    # start_time = time.time()
    # trained_svm = train_test_svm(X, y)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("Elapsed time:", elapsed_time, "seconds")
    # print('-------------------------------')

def Train_withpca():
    # Read the data from the CSV file
    data = pd.read_csv("pca_results.csv")

    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']

    # Assuming 'X' contains features and 'y' contains the target variable of the dataset
    # Replace 'X' and 'y' with your actual features and target variable
    start_time = time.time()
    trained_rf = train_test_random_forest(X, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    print('-------------------------------')

    
    start_time = time.time()
    trained_gb = train_test_gradient_boosting(X, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    print('-------------------------------')
    start_time = time.time()
    trained_lr = train_test_logistic_regression(X, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    print('-------------------------------')
    # start_time = time.time()
    # trained_svm = train_test_svm(X, y)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("Elapsed time:", elapsed_time, "seconds")
    # print('-------------------------------')


Train()

import random
def detect_failure(data_point):
    # Unpack the data point
    air_temperature, process_temperature, rotational_speed, torque, tool_wear = data_point[1:]

    # Tool Wear Failure (TWF)
    if 200 <= tool_wear <= 240:
        return "Tool Wear Failure (TWF)"

    # Heat Dissipation Failure (HDF)
    if abs(air_temperature - process_temperature) < 8.6 and rotational_speed < 1380:
        return "Heat Dissipation Failure (HDF)"

    # Power Failure (PWF)
    power = torque * (rotational_speed / 9.55)  # Convert rpm to rad/s
    if power < 3500 or power > 9000:
        return "Power Failure (PWF)"

    # Overstrain Failure (OSF)
    if tool_wear * torque > 11000:  # Assuming L product variant
        return "Overstrain Failure (OSF)"

    # Random Failure (RNF)
    # Since this is random, we don't need to check any conditions
    if random.random() < 0.001:
        return "Random Failure (RNF)"

    return "No Failure Detected"

def classify(datapoints):
    # Load the trained model
    model = joblib.load('models/random_forest_model.pkl')
    # Make predictions for each datapoint
    predictions = model.predict(datapoints)
    print(predictions)
    print(list(map(detect_failure, datapoints)))


# classify([[0,301.8274080716954,311.15288968603454,1442,45.293782959224345,208]
# ,[0,296.6,307.4,1521,38.6,173]
# ,[0,297.9618222375274,308.2945460339325,1404,56.009111187636975,217]
# ,[1,300.1,309.1,1731,30.4,92]])

