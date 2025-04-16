# Medical Insurance Cost Prediction – Linear Regression Analysis

This project explores the application of linear regression with regularization to predict medical insurance charges based on personal and demographic features. 
The dataset used contains attributes such as age, sex, BMI, number of children, smoking status, and region.

---

## 📂 Dataset

The dataset (`insurData.csv`) includes the following columns:

- `age`
- `sex` (male/female)
- `bmi`
- `children`
- `smoker` (yes/no)
- `region` (southeast, southwest, northeast, northwest)
- `charges` (target variable, normalized by dividing by 1000)

---

## ⚙️ Features of the Project

- **Preprocessing:**
  - One-hot encoding for the `region` feature.
  - Binary encoding for `smoker` and `sex`.
  - Normalization of the `charges` column.
  - Addition of a bias term (intercept).

- **Modeling:**
  - Baseline model that predicts a constant (mean of charges).
  - Linear regression using least squares with Tikhonov (L2) regularization.

- **Evaluation:**
  - MSE (Mean Squared Error) calculated for both training and test data.
  - Relative MSE with respect to the baseline.
  - 10 randomized experiments with different train/test splits (80/20 split).

- **Experiment Variants:**
  - First experiment: uses all features including categorical ones.
  - Second experiment: uses only numerical features (excluding smoker and region).

---

## 📊 Results

The model prints detailed evaluation results per experiment, including:

- Absolute and relative MSE for both training and test sets.
- Effectiveness of feature selection on model accuracy.

---

## 🧪 Running the Project

Make sure you have the following dependencies installed:

pip install pandas numpy scikit-learn matplotlib
Place the insurData.csv file in the designated path defined in main().
Then run the script: python your_script.py

📁 Output

Updated dataset with one-hot and binary encoding saved as updated_insurData.csv.

Printed experiment results for both versions (with and without categorical features).

🧠 Author Notes

This project showcases how preprocessing and feature selection can significantly affect the performance of a linear regression model. Regularization is also applied to ensure numerical stability during matrix inversion and to prevent overfitting.

📜 License

This project is for educational purposes and free to use and modify.




