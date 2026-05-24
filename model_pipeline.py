import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import shap
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=2500):
    """
    Generates a realistic synthetic credit risk dataset with causal links and fairness biases.
    
    Causal Graph structure:
    - Gender (sensitive, immutable)
    - Race (sensitive, immutable)
    - Age (immutable)
    - EmploymentLength = f(Age) + noise
    - AnnualIncome = f(Age, EmploymentLength, Gender, Race) + noise (represents wage gap bias)
    - RequestedAmount = f(AnnualIncome) + noise
    - MonthlyDebt = f(AnnualIncome) + noise
    - DTI = MonthlyDebt / (AnnualIncome / 12)
    - CreditScore = f(Age, DTI, EmploymentLength) + noise
    - PriorDefaults = f(CreditScore) + noise
    - LoanToIncome = RequestedAmount / AnnualIncome
    - Approved = f(CreditScore, DTI, PriorDefaults, LoanToIncome, Gender, Race) + noise (biased ground truth)
    """
    
    # 1. Sensitive & Immutable attributes
    gender = np.random.binomial(1, 0.48, n_samples)  # 0: Male, 1: Female
    race = np.random.binomial(1, 0.35, n_samples)    # 0: White, 1: Minority
    age = np.random.randint(21, 68, n_samples)
    
    # 2. Employment Length (depends on age)
    # Expected employment length increases with age, capped at (age - 18)
    emp_length = np.zeros(n_samples)
    for i in range(n_samples):
        max_emp = max(0, age[i] - 18)
        emp_length[i] = int(np.clip(np.random.normal(max_emp * 0.4, 4), 0, max_emp))
        
    # 3. Annual Income (depends on age, employment length, gender, race for systemic biases)
    # Base income starts at $28k, increases with age & employment, with simulated systemic wage gaps
    base_income = 30000
    income_age_effect = (age - 21) * 800
    income_emp_effect = emp_length * 1500
    
    # Introduce systemic bias in income (wage gaps)
    gender_wage_gap = -6000 * gender
    race_wage_gap = -8000 * race
    
    income_noise = np.random.normal(0, 12000, n_samples)
    income = base_income + income_age_effect + income_emp_effect + gender_wage_gap + race_wage_gap + income_noise
    income = np.clip(income, 18000, 240000).astype(int)
    
    # 4. Requested Loan Amount (correlated with income)
    # People with higher income ask for larger loans
    requested_amount = income * np.random.uniform(0.15, 0.65, n_samples)
    requested_amount = np.clip(requested_amount, 5000, 150000).round(-2)
    
    # 5. Monthly Debt (excluding the new loan)
    # Correlated with income but wide spread
    monthly_debt = (income / 12) * np.clip(np.random.normal(0.25, 0.12, n_samples), 0.05, 0.70)
    monthly_debt = monthly_debt.round(-1)
    
    # 6. Debt-to-Income (DTI) Ratio (computed)
    dti = (monthly_debt * 12) / income
    
    # 7. Credit Score (depends on age, DTI, employment length)
    # Credit score increases with age/stability, decreases with high DTI
    base_credit = 550
    credit_age_effect = (age - 21) * 2.5
    credit_emp_effect = emp_length * 3.5
    credit_dti_effect = -150 * dti
    credit_noise = np.random.normal(0, 45, n_samples)
    
    credit_score = base_credit + credit_age_effect + credit_emp_effect + credit_dti_effect + credit_noise
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    
    # 8. Prior Defaults (higher default rates for low credit scores)
    # Default prob is high if credit < 580, very low if credit > 720
    default_prob = 1 / (1 + np.exp((credit_score - 560) / 45))
    prior_defaults = np.random.binomial(1, default_prob)
    
    # 9. Loan-to-Income Ratio (computed)
    loan_to_income = requested_amount / income
    
    # Create DataFrame
    df = pd.DataFrame({
        'Gender': gender,
        'Race': race,
        'Age': age,
        'EmploymentLength': emp_length,
        'AnnualIncome': income,
        'MonthlyDebt': monthly_debt,
        'RequestedAmount': requested_amount,
        'CreditScore': credit_score,
        'PriorDefaults': prior_defaults,
        'DTI': dti,
        'LoanToIncome': loan_to_income
    })
    
    # 10. Approval Decision (Ground Truth with historical bias)
    # Higher credit, lower DTI, no prior defaults, lower loan-to-income increase approval probability
    log_odds = (
        0.022 * credit_score 
        - 4.2 * dti 
        - 2.8 * prior_defaults 
        - 1.8 * loan_to_income 
        + 0.08 * emp_length
        - 0.4 * gender  # Historical bias against females
        - 0.5 * race    # Historical bias against minorities
        - 5.8           # Intercept to balance approved/rejected rates
    )
    
    prob_approved = 1 / (1 + np.exp(-log_odds))
    approved = np.random.binomial(1, prob_approved)
    df['Approved'] = approved
    
    return df

def compute_reweighting_weights(df, sensitive_col, target_col):
    """
    Computes sample weights using the Kamiran & Calders reweighting method.
    Balances the joint distribution of sensitive attribute and target variable.
    """
    n = len(df)
    n_s0 = len(df[df[sensitive_col] == 0])
    n_s1 = len(df[df[sensitive_col] == 1])
    n_y0 = len(df[df[target_col] == 0])
    n_y1 = len(df[df[target_col] == 1])
    
    n_s0_y0 = len(df[(df[sensitive_col] == 0) & (df[target_col] == 0)])
    n_s0_y1 = len(df[(df[sensitive_col] == 0) & (df[target_col] == 1)])
    n_s1_y0 = len(df[(df[sensitive_col] == 1) & (df[target_col] == 0)])
    n_s1_y1 = len(df[(df[sensitive_col] == 1) & (df[target_col] == 1)])
    
    weights = np.ones(n)
    
    # Avoid division by zero
    w_s0_y0 = (n_s0 * n_y0) / (n * n_s0_y0) if n_s0_y0 > 0 else 1.0
    w_s0_y1 = (n_s0 * n_y1) / (n * n_s0_y1) if n_s0_y1 > 0 else 1.0
    w_s1_y0 = (n_s1 * n_y0) / (n * n_s1_y0) if n_s1_y0 > 0 else 1.0
    w_s1_y1 = (n_s1 * n_y1) / (n * n_s1_y1) if n_s1_y1 > 0 else 1.0
    
    weights[(df[sensitive_col] == 0) & (df[target_col] == 0)] = w_s0_y0
    weights[(df[sensitive_col] == 0) & (df[target_col] == 1)] = w_s0_y1
    weights[(df[sensitive_col] == 1) & (df[target_col] == 0)] = w_s1_y0
    weights[(df[sensitive_col] == 1) & (df[target_col] == 1)] = w_s1_y1
    
    return weights

def train_and_save_pipeline():
    """
    Trains standard and debiased models and saves all artifacts.
    """
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    # Save the raw data
    data_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(data_dir, "synthetic_loan_data.csv"), index=False)
    
    # Define features (we exclude sensitive attributes directly from model features
    # to comply with typical anti-discrimination law, though bias persists via proxies)
    features = [
        'Age', 'EmploymentLength', 'AnnualIncome', 
        'MonthlyDebt', 'RequestedAmount', 'CreditScore', 
        'PriorDefaults', 'DTI', 'LoanToIncome'
    ]
    
    X = df[features]
    y = df['Approved']
    
    # Split training and testing sets, keeping sensitive columns in test for auditing
    X_train, X_test, y_train, y_test, gender_train, gender_test, race_train, race_test = train_test_split(
        X, y, df['Gender'], df['Race'], test_size=0.25, random_state=42, stratify=y
    )
    
    # Build test dataframe for auditing
    test_df = X_test.copy()
    test_df['Gender'] = gender_test
    test_df['Race'] = race_test
    test_df['Approved'] = y_test
    test_df.to_csv(os.path.join(data_dir, "test_data.csv"), index=False)
    
    print("Training standard XGBoost model...")
    # Standard Model
    model_std = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    model_std.fit(X_train, y_train)
    
    # Evaluate Standard Model
    y_pred_std = model_std.predict(X_test)
    acc_std = accuracy_score(y_test, y_pred_std)
    f1_std = f1_score(y_test, y_pred_std)
    print(f"Standard Model - Accuracy: {acc_std:.4f}, F1: {f1_std:.4f}")
    
    # Train Fair Model (Reweighted based on Gender)
    print("Computing re-weighting sample weights...")
    train_df = X_train.copy()
    train_df['Gender'] = gender_train
    train_df['Approved'] = y_train
    
    sample_weights = compute_reweighting_weights(train_df, 'Gender', 'Approved')
    
    print("Training fairness-mitigated XGBoost model...")
    model_fair = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    model_fair.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate Fair Model
    y_pred_fair = model_fair.predict(X_test)
    acc_fair = accuracy_score(y_test, y_pred_fair)
    f1_fair = f1_score(y_test, y_pred_fair)
    print(f"Fair Model - Accuracy: {acc_fair:.4f}, F1: {f1_fair:.4f}")
    
    # Create SHAP Explainer (we use the standard model for our recourse and explanation UI)
    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model_std)
    
    # Save the artifacts
    artifacts = {
        'model_std': model_std,
        'model_fair': model_fair,
        'explainer': explainer,
        'features': features,
        'stats': {
            'acc_std': acc_std,
            'f1_std': f1_std,
            'acc_fair': acc_fair,
            'f1_fair': f1_fair
        }
    }
    
    with open(os.path.join(data_dir, "model_artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
        
    print("All artifacts successfully saved to model_artifacts.pkl!")

if __name__ == "__main__":
    train_and_save_pipeline()
