"""
Script: LR_angle_models.py

This script loads training and test data from JSON files, extracts individual absolute error values
from the 'best_window' as predictors, and uses these predictors to train and evaluate linear regression
models to predict the tilt angle. The script generates performance metrics and visualizes the results
with plots.

Models:
1. Using the sum of absolute error values from the 'best_window' as a single predictor.
2. Using each individual absolute error value from the 'best_window' as separate predictors.

Functions:
- load_json: Load JSON data from a file.
- extract_absolute_errors: Extract absolute error values from the 'best_window' field in the data.
- extract_sum_of_errors: Extract the sum of absolute error values from the 'best_window' field in the data.
- create_and_evaluate_model: Train and evaluate a linear regression model, generate metrics, and create plots.
- create_eda_plot: Generate an exploratory data analysis scatter plot.
- main: Main function to orchestrate data loading, feature extraction, model training, evaluation, and plotting.

Usage:
    python LR_angle_models.py --train path_to_train_json --test path_to_test_json --output path_to_output_directory

Command-line Arguments:
    --train : Path to the training JSON file.
    --test : Path to the test JSON file.
    --output : Path to the output directory for saving plots and metrics.

Example:
    python LR_angle_models.py --train train_data.json --test test_data.json --output results/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import os

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def extract_absolute_errors(data):
    absolute_errors = [item['best_window']['absolute_error'] for item in data]
    return np.array(absolute_errors)

def extract_sum_of_errors(data):
    sum_of_errors = [sum(item['best_window']['absolute_error']) for item in data]
    return np.array(sum_of_errors).reshape(-1, 1)

def create_and_evaluate_model(X_train, y_train, X_test, y_test, feature_name, log_transform, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)

    metrics = {
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "R-squared (R2)": r2
    }

    metrics_filename = os.path.join(output_dir, f"{feature_name}_{'log' if log_transform else 'nonlog'}_metrics.json")
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Plotting
    plt.figure(figsize=(14, 6))

    # Plot 1: Predicted vs Actual Tilt Angles (Train and Test)
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.7, color='blue', label='Train')
    plt.scatter(y_test, y_test_pred, alpha=0.7, color='green', label='Test')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
    plt.title(f'Predicted vs Actual Tilt Angles ({feature_name}{" (Log)" if log_transform else ""})')
    plt.xlabel('Actual Tilt Angle')
    plt.ylabel('Predicted Tilt Angle')
    plt.legend()
    plt.grid(True)

    # Plot 2: Residuals (Test)
    plt.subplot(1, 2, 2)
    residuals = y_test - y_test_pred
    plt.scatter(y_test, residuals, alpha=0.7, color='green')
    plt.hlines(0, min(y_test), max(y_test), colors='red', linestyles='--')
    plt.title(f'Residuals ({feature_name}{" (Log)" if log_transform else ""})')
    plt.xlabel('Actual Tilt Angle')
    plt.ylabel('Residual')
    plt.grid(True)

    plot_filename = os.path.join(output_dir, f"{feature_name}_{'log' if log_transform else 'nonlog'}_plot.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

def create_eda_plot(X_train, y_train, X_test, y_test, feature_name, log_transform, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(18, 6))

    for i in range(X_train.shape[1]):
        plt.subplot(1, 3, i+1)
        plt.scatter(X_train[:, i], y_train, alpha=0.7, color='blue', label=f'Train - Predictor {i+1}')
        plt.scatter(X_test[:, i], y_test, alpha=0.7, color='green', label=f'Test - Predictor {i+1}')
        plt.title(f'Predictor {i+1} vs Tilt Angle')
        plt.xlabel(f'{feature_name} {i+1}{" (Log)" if log_transform else ""}')
        plt.ylabel('Tilt Angle')
        plt.legend()
        plt.grid(True)

    plot_filename = os.path.join(output_dir, f"eda_{feature_name}_{'log' if log_transform else 'nonlog'}_combined_plot.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

def main(train_file, test_file, output_dir):
    train_data = load_json(train_file)
    test_data = load_json(test_file)

    # Model using the sum of absolute errors as a single predictor
    feature_name = "sum_of_absolute_errors"

    X_train_sum = extract_sum_of_errors(train_data)
    y_train = np.array([item['tilt_angle'] for item in train_data])

    X_test_sum = extract_sum_of_errors(test_data)
    y_test = np.array([item['tilt_angle'] for item in test_data])

    create_and_evaluate_model(X_train_sum, y_train, X_test_sum, y_test, feature_name, False, output_dir)
    create_eda_plot(X_train_sum, y_train, X_test_sum, y_test, feature_name, False, output_dir)

    # Model using individual absolute errors as separate predictors
    feature_name = "individual_absolute_errors"

    X_train_individual = extract_absolute_errors(train_data)
    y_train = np.array([item['tilt_angle'] for item in train_data])

    X_test_individual = extract_absolute_errors(test_data)
    y_test = np.array([item['tilt_angle'] for item in test_data])

    create_and_evaluate_model(X_train_individual, y_train, X_test_individual, y_test, feature_name, False, output_dir)
    create_eda_plot(X_train_individual, y_train, X_test_individual, y_test, feature_name, False, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Regression to predict tilt angle using absolute error predictors.")
    parser.add_argument('--train', type=str, required=True, help='Path to the training JSON file.')
    parser.add_argument('--test', type=str, required=True, help='Path to the test JSON file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory for plots and metrics.')

    args = parser.parse_args()

    main(args.train, args.test, args.output)
