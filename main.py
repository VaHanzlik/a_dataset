# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:04:11 2023

@author: vhanzlik
"""
import pandas as pd
import data_processing as dp


def main():
   
    # Load data
    candidates_path = "data\\candidates.csv"
    admi_path = "data\\admissions.csv"
    candidates, admi = dp.load_data(candidates_path, admi_path)

    # Join data
    data = dp.join_datasets(candidates, admi)

    # Check data quality
    data_quality_report = dp.data_quality(data)
    print("\nData Quality Report:")
    for key, value in data_quality_report.items():
        print(f"{key}: {value}")
    dp.plot_data_quality(data_quality_report)

    # clear data
    # drop duplicates
    data = data.drop_duplicates(subset = 'candidate_id')
    # drop rows where age is outside of bounds
    data = data.loc[(data['age'] >= 18) & (data['age'] <= 99)]
    
    # Check data quality
    data_quality_report = dp.data_quality(data)
    print("\nData Quality Report after cleaning the data:")
    for key, value in data_quality_report.items():
        print(f"{key}: {value}")
    dp.plot_data_quality(data_quality_report)
    # Calculate average age
    avg_age = dp.average_age(data)
    print("\n")
    print(f"The average age of candidates is: {avg_age:.2f}")
        
    # Perform feature selection
    selected_data = data[['age',
                        'gender',
                        'no_children',
                        'department',
                        'admit']].copy()
    # map "admitted" to 1 and "rejected" to 0
    selected_data.loc[:, 'admit_numeric'] = selected_data['admit'].map({'Admitted': 1, 'Rejected': 0})
    selected_data = selected_data.drop('admit', axis=1)
    # Prepare data
    prepared_data = pd.get_dummies(selected_data, columns=['gender', 'department'])
    prepared_data['no_children'] = prepared_data['no_children'].fillna(prepared_data['no_children'].mean())
    
    y = prepared_data['admit_numeric']
    X = prepared_data.drop('admit_numeric', axis=1)

   
    # Train and evaluate the model
    model, X_test, y_test = dp.train_model(X, y)
    accuracy, confusion_matrix = dp.evaluate_model(model, X_test, y_test)
    labels = ['Admitted','Rejected']
    print(f"\nModel accuracy: {accuracy:.2f}")
    print("\n")
    dp.plot_confusion_matrix(confusion_matrix, labels)


    # Check for discrimination
    female_rate, male_rate, chi2, p = dp.check_discrimination(data, gender_col = 'gender', admit_col = 'admit')
    # print results
    print("Check for discrimination")
    print(f"Female admission rate: {female_rate:.2f}")
    print(f"Male admission rate: {male_rate:.2f}")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"p-value: {p:.5f}")
    
   

if __name__ == "__main__":
    main()