import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
def generate_data(N, M, input_type, output_type):    
    if input_type == "real":
        X = pd.DataFrame(np.random.randn(N, M))
    elif input_type == "discrete":
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})

    if output_type == "real":
        y = pd.Series(np.random.randn(N))
    elif output_type == "discrete":
        y = pd.Series(np.random.randint(M, size=N), dtype="category")

    return X, y



# Function to calculate average time (and std) taken by fit() and predict() for different N and M for 4 different cases of DTs

def evaluate_runtime(N, M, input_type, output_type, test_size, criterias, num_average_time):
    X, y = generate_data(N, M, input_type, output_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    time_data = {}
    for criteria in criterias:
        fit_times = []
        predict_times = []
        for _ in range(num_average_time):

            tree = DecisionTree(criterion=criteria, max_depth=5)
            start_train = time.time()
            tree.fit(X_train, y_train)
            fit_times.append(time.time() - start_train)

            start_test = time.time()
            y_pred = tree.predict(X_test)
            predict_times.append(time.time() - start_test)

        avg_train_time = np.mean(fit_times)
        avg_test_time = np.mean(predict_times)

        std_train_time = np.std(fit_times)
        std_test_time = np.std(predict_times)

        print(f"    Criteria: {criteria}")
        print(f"        Average Training Time: {avg_train_time:.4f} seconds (std: {std_train_time:.4f})")
        print(f"        Average Prediction Time: {avg_test_time:.4f} seconds (std: {std_test_time:.4f})")

        time_data[criteria] = {
            "train_time": avg_train_time,
            "test_time": avg_test_time,
            "std_train_time": std_train_time,
            "std_test_time": std_test_time
        }

    return time_data


def run_n_m(N_values, M_values, input_types, output_types, test_size, criterias, num_average_time):
    results = {}
    for N in N_values:
        for M in M_values:
            print(f"\nEvaluating for N={N}, M={M}\n")
            results[(N, M)] = {}
            for input_type in input_types:
                for output_type in output_types:
                    print(f"    Input Type: {input_type}, Output Type: {output_type}")
                    single_data = evaluate_runtime(N, M, input_type, output_type, test_size, criterias, num_average_time)
                    results[(N, M)][(input_type, output_type)] = single_data
                    print()
            print("=" * 50)
    return results


# Function to plot the results
def plot_time_complexity_separate(results, N_values, M_values, criteria):
    # Plot training and prediction time vs N values, keeping M constant
    print("Time vs Number of Samples (N)")
    plt.figure(figsize=(14, 5*len(M_values)))
    plt.suptitle(f"Time vs Number of Samples (N) - Criteria: {criteria}", y=1, fontsize=16)

    for i, M in enumerate(M_values):
        ax1 = plt.subplot(len(M_values), 2, 2*i + 1)
        ax2 = plt.subplot(len(M_values), 2, 2*i + 2)

        for input_type in ["discrete", "real"]:
            for output_type in ["discrete", "real"]:
                train_times = []
                prediction_times = []

                for N in N_values:
                    data = results[(N, M)][(input_type, output_type)][criteria]
                    train_times.append(data["train_time"])
                    prediction_times.append(data["test_time"])
                
                ax1.plot(N_values, train_times, marker='o', label=f'{input_type.capitalize()}-{output_type.capitalize()}')
                ax2.plot(N_values, prediction_times, marker='o', label=f'{input_type.capitalize()}-{output_type.capitalize()}')

        ax1.set_xlabel("Number of Samples (N)")
        ax1.set_ylabel("Training Time (seconds)")
        ax1.set_title(f"Training Time vs Number of Samples (N), M = {M}")
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel("Number of Samples (N)")
        ax2.set_ylabel("Prediction Time (seconds)")
        ax2.set_title(f"Prediction Time vs Number of Samples (N), M = {M}")
        ax2.legend()
        ax2.grid(True)

    plt.savefig(f"./Task-5 Decision Tree Implementation/5.4 Data/5.4_time_vs_N_{criteria}.png", bbox_inches='tight', dpi = 300)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    plt.show()

    print("=" * 50)

    # Plot training and prediction time vs M values, keeping N constant
    print("Time vs Number of Features (M)")
    plt.figure(figsize=(14, 5*len(N_values)))
    plt.suptitle(f"Time vs Number of Features (M) - Criteria: {criteria}", y=1, fontsize=16)

    for i, N in enumerate(N_values):
        ax1 = plt.subplot(len(N_values), 2, 2*i + 1)
        ax2 = plt.subplot(len(N_values), 2, 2*i + 2)

        for input_type in ["discrete", "real"]:
            for output_type in ["discrete", "real"]:
                train_times = []
                prediction_times = []
                
                for M in M_values:
                    data = results[(N, M)][(input_type, output_type)][criteria]
                    train_times.append(data["train_time"])
                    prediction_times.append(data["test_time"])

                ax1.plot(M_values, train_times, marker='o', label=f'{input_type.capitalize()}-{output_type.capitalize()}')
                ax2.plot(M_values, prediction_times, marker='o', label=f'{input_type.capitalize()}-{output_type.capitalize()}')

        ax1.set_xlabel("Number of Features (M)")
        ax1.set_ylabel("Training Time (seconds)")
        ax1.set_title(f"Training Time vs Number of Features (M), N = {N}")
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel("Number of Features (M)")
        ax2.set_ylabel("Prediction Time (seconds)")
        ax2.set_title(f"Prediction Time vs Number of Features (M), N = {N}")
        ax2.legend()
        ax2.grid(True)

    plt.savefig(f"./Task-5 Decision Tree Implementation/5.4 Data/5.4_time_vs_M_{criteria}.png", bbox_inches='tight', dpi = 300)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    plt.show()


# Run the functions, Learn the DTs and Show the results/plots

N_values = [50, 100, 500, 1000, 5000]
M_values = [1, 5, 10, 20, 50, 100]
criterias = ["information_gain", "gini_index"]
input_types = ["real", "discrete"]
output_types = ["real", "discrete"]
test_size = 0.3

results = run_n_m(N_values, M_values, input_types, output_types, test_size, criterias, num_average_time)


# Save the results to a file
import pickle
if not os.path.exists(r'./Task-5 Decision Tree Implementation/5.4 Data'):
    os.makedirs(r'./Task-5 Decision Tree Implementation/5.4 Data')
with open(r'./Task-5 Decision Tree Implementation/5.4 Data/5.4_results.pkl', 'wb') as f:
    pickle.dump(results, f)


# Plot the results
plot_time_complexity_separate(results, N_values, M_values, criteria="information_gain")
plot_time_complexity_separate(results, N_values, M_values, criteria="gini_index")