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

def extract_features(data, feature_name, log_transform=False):
    if feature_name in ["total_absolute_error", "total_squared_error"]:
        errors = [item['best_window'][feature_name] for item in data]
    elif feature_name == "frequency_difference":
        errors = [abs(item['predicted_frequency'] - item['ground_truth_frequency']) for item in data]
    elif feature_name == "frequency_squared_error":
        errors = [(item['predicted_frequency'] - item['ground_truth_frequency']) ** 2 for item in data]
    else:
        raise ValueError(f"Unknown feature name: {feature_name}")
    
    if log_transform:
        return np.log1p(errors)
    else:
        return errors

def extract_multi_features(data, log_transform=False, squared=False):
    errors = [item['best_window']['absolute_error'] for item in data]
    features = np.array(errors)
    if log_transform:
        features = np.log1p(features)
    if squared:
        features = features ** 2
    return features

def create_and_evaluate_model(X_train, y_train, X_test, y_test, feature_name, log_transform, output_dir):
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
    plt.figure(figsize=(14, 6))
    plt.scatter(X_train, y_train, alpha=0.7, color='blue', label='Train')
    plt.scatter(X_test, y_test, alpha=0.7, color='green', label='Test')
    plt.title(f'EDA Scatter Plot ({feature_name}{" (Log)" if log_transform else ""})')
    plt.xlabel(f'{feature_name}{" (Log)" if log_transform else ""}')
    plt.ylabel('Tilt Angle')
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(output_dir, f"eda_{feature_name}_{'log' if log_transform else 'nonlog'}_plot.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

def main(train_file, test_file, output_dir):
    train_data = load_json(train_file)
    test_data = load_json(test_file)

    features = ["total_absolute_error", "total_squared_error", "frequency_difference", "frequency_squared_error"]

    single_pred_dir = os.path.join(output_dir, "single_pred")
    multi_pred_dir = os.path.join(output_dir, "multi_pred")
    os.makedirs(single_pred_dir, exist_ok=True)
    os.makedirs(multi_pred_dir, exist_ok=True)
    
    # Single Predictor Models
    for feature in features:
        for log_transform in [False, True]:
            X_train = np.array(extract_features(train_data, feature, log_transform)).reshape(-1, 1)
            y_train = np.array([item['tilt_angle'] for item in train_data])

            X_test = np.array(extract_features(test_data, feature, log_transform)).reshape(-1, 1)
            y_test = np.array([item['tilt_angle'] for item in test_data])

            create_and_evaluate_model(X_train, y_train, X_test, y_test, feature, log_transform, single_pred_dir)
            create_eda_plot(X_train, y_train, X_test, y_test, feature, log_transform, single_pred_dir)

    # Multi Predictor Models
    for log_transform in [False, True]:
        for squared in [False, True]:
            X_train = extract_multi_features(train_data, log_transform, squared)
            y_train = np.array([item['tilt_angle'] for item in train_data])

            X_test = extract_multi_features(test_data, log_transform, squared)
            y_test = np.array([item['tilt_angle'] for item in test_data])

            feature_name = f"multi_pred_{'log_' if log_transform else ''}{'squared_' if squared else ''}features"
            create_and_evaluate_model(X_train, y_train, X_test, y_test, feature_name, False, multi_pred_dir)
            create_eda_plot(X_train, y_train, X_test, y_test, feature_name, log_transform, multi_pred_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Regression to predict tilt angle from various features.")
    parser.add_argument('--train', type=str, required=True, help='Path to the training JSON file.')
    parser.add_argument('--test', type=str, required=True, help='Path to the test JSON file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory for plots and metrics.')

    args = parser.parse_args()

    main(args.train, args.test, args.output)
