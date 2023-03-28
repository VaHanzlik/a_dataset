# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:02:56 2023

@author: vhanzlik
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np

def plot_data_quality(quality_params):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(quality_params.keys(), quality_params.values())
    plt.xticks(rotation=45)
    plt.title('Data Quality Parameters')
    plt.show()

def plot_confusion_matrix(confusion_matrix, labels):
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def load_data(candidates_path, admit_path):
    """
    Load candidates and admit datasets from CSV files.

    Args:
        candidates_path (str): Path to the candidates CSV file.
        admit_path (str): Path to the admit CSV file.

    Returns:
        candidates (pd.DataFrame): DataFrame containing candidates data.
        admit (pd.DataFrame): DataFrame containing admit data.
    """
    candidates = pd.read_csv(candidates_path)
    admit = pd.read_csv(admit_path)
    return candidates, admit

def average_age(candidates):
    """
   Calculate the average age of the candidates.

   Args:
       candidates (pd.DataFrame): DataFrame containing candidates data.

   Returns:
       float: The average age of the candidates.
   """
   
    return candidates['age'].mean()

def data_quality(candidates):
    """
    Analyze the data quality of the candidates DataFrame.

    Args:
        candidates (pd.DataFrame): DataFrame containing candidates data.

    Returns:
        dict: Dictionary containing various data quality parameters.
    """
    quality_params = {
        "missing_age": candidates['age'].isna().sum(),
        "missing_gender": candidates['gender'].isna().sum(),
        "missing_no_children": candidates['no_children'].isna().sum(),
        "missing_emp_card_id": candidates['emp_card_id'].isna().sum(),
        "duplicate_candidate_ids": candidates['candidate_id'].astype(int).duplicated().sum(),
        "min_age": candidates['age'].min(),
        "max_age": candidates['age'].max(),
        "num_unique_genders": candidates['gender'].nunique(),
        "num_unique_no_children": candidates['no_children'].nunique(),
        "num_unique_emp_card_id": candidates['emp_card_id'].nunique(),
        "sum_of_admitted": candidates['admit'].value_counts().get('Admitted'),
        "sum_of_rejected": candidates['admit'].value_counts().get('Rejected')

    }
    return quality_params

def join_datasets(candidates, admit):
    """
    Join the candidates and admit DataFrames using candidate_id as the key.

    Args:
        candidates (pd.DataFrame): DataFrame containing candidates data.
        admit (pd.DataFrame): DataFrame containing admit data.

    Returns:
        pd.DataFrame: DataFrame containing the joined data.
    """
    
    return pd.merge(candidates, admit, on="candidate_id")


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train a RandomForestClassifier model and split the dataset into training and testing sets.

    Args:
        X (pd.DataFrame): DataFrame containing the features.
        y (pd.Series): Series containing the labels.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Seed used by the random number generator. Defaults to 42.

    Returns:
        tuple: Trained model, X_test, and y_test data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the trained model using a chosen metric.

    Args:
        model (RandomForestClassifier): Trained RandomForestClassifier instance.
        X_test (pd.DataFrame): DataFrame containing the testing features.
        y_test (pd.Series): Series containing the testing labels.

    Returns:
        float: The evaluation metric value (accuracy).
    """
    
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)

def check_discrimination(df, gender_col, admit_col):
    '''
    Check for discrimination based on gender in admission data.
    
    Parameters:
        df (DataFrame): Input data.
        gender_col (str): Name of the column containing gender data.
        admit_col (str): Name of the column containing admission data (admitted or rejected).
    
    Returns:
        (float, float, float, float): Tuple containing the female admission rate, male admission rate,
                                      chi-square statistic, and p-value for the chi-square test.
    '''
    # calculate admission rates for each gender
    female_admitted = df[(df[gender_col] == 'Female') & (df[admit_col] == 'Admitted')].shape[0]
    female_rejected = df[(df[gender_col] == 'Female') & (df[admit_col] == 'Rejected')].shape[0]
    male_admitted = df[(df[gender_col] == 'Male') & (df[admit_col] == 'Admitted')].shape[0]
    male_rejected = df[(df[gender_col] == 'Male') & (df[admit_col] == 'Rejected')].shape[0]
    
    female_admission_rate = female_admitted / (female_admitted + female_rejected)
    male_admission_rate = male_admitted / (male_admitted + male_rejected)
    
    # perform chi-square test for independence
    contingency_table = np.array([[female_admitted, female_rejected], [male_admitted, male_rejected]])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    return female_admission_rate, male_admission_rate, chi2, p