from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt

# Load CSV file, normalize values, encode categorical variables
def preprocess_data(path: str, filename: str = 'insurData.csv'):
    df = pd.read_csv(path + filename)
    
    # Add bias column (alpha_0)
    df.insert(0, "Ones", 1)

    # Convert units (to reduce scale)
    df["charges"] = df["charges"] / 1000

    # Manual one-hot encoding for 'region'
    df.insert(7, "southeast", 0)
    df.insert(8, "southwest", 0)
    df.insert(9, "northeast", 0)
    df.insert(10, "northwest", 0)

    for i in range(len(df['region'])):
        if df['region'][i] == 'southeast':
            df.at[i, 'southeast'] = 1
        elif df['region'][i] == 'southwest':
            df.at[i, 'southwest'] = 1
        elif df['region'][i] == 'northeast':
            df.at[i, 'northeast'] = 1
        elif df['region'][i] == 'northwest':
            df.at[i, 'northwest'] = 1
    df.drop(['region'], axis=1, inplace=True)

    # Binary encoding for smoker and sex columns
    df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)

# Compute MSE for constant model: alpha_0 = mean of charges
def compute_baseline_mse(y: np.ndarray) -> float:
    a0 = np.mean(y)
    mse0 = np.mean((a0 - y) ** 2)
    return a0, mse0

# Solve least squares with regularization (Tikhonov)
def compute_regularized_least_squares(X, y, lamb=1e-5):
    n = X.shape[1]
    xtx = X.T @ X
    reg = lamb * np.identity(n) # Prevents singular matrices
    x_hat = np.linalg.inv(xtx + reg) @ X.T @ y
    return x_hat

# Compute MSE between original y and predicted y_hat
def evaluate_model(X, y, x_hat):
    y_pred = X @ x_hat
    mse = np.mean((y_pred - y) ** 2)
    return mse

# Run 10 randomized experiments with different train/test splits
def run_experiment_q5(df: pd.DataFrame, a0: float, mse0: float):
    X = df.drop('charges', axis=1).values
    y = df['charges'].values

    results = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=i)
        x_hat = compute_regularized_least_squares(X_train, y_train, lamb=1e-5)

        mse_train = evaluate_model(X_train, y_train, x_hat)
        mse_test = evaluate_model(X_test, y_test, x_hat)

        rel_train = mse_train / mse0
        rel_test = mse_test / mse0

        results.append([i+1, mse_train, rel_train, mse_test, rel_test])

    print("=== Results for Question 5: With all features ===")
    print("Run | Train MSE | Train MSE/MSE₀ | Test MSE | Test MSE/MSE₀")
    print("----|-----------|----------------|----------|----------------")
    for row in results:
        print(f"{row[0]:>3} | {row[1]:>9.4f} | {row[2]:>14.4f} | {row[3]:>8.4f} | {row[4]:>14.4f}")
    print("")

def main():
    path = r"C:\Users\JV\Downloads\\"
    df = preprocess_data(path)
    df.to_csv(path + 'updated_insurData.csv', index=False) # Save updated version of the dataset

    y = df['charges'].values
    a0, mse0 = compute_baseline_mse(y)

    print(f"Baseline α₀: {a0:.4f}")
    print(f"Baseline MSE₀: {mse0:.4f}\n")

    run_experiment_q5(df, a0, mse0) # calculate for 5c
    
    df.drop(['smoker'], axis=1, inplace=True)
    df.drop(['southeast', 'southwest', 'northeast', 'northwest'], axis=1, inplace=True)

    run_experiment_q5(df, a0, mse0) # calculate for 5d

if __name__ == "__main__":
    main()
