import joblib
import pandas as pd
import numpy as np

# Load artifacts
model = joblib.load("model.joblib")
explainer = joblib.load("shap_explainer.joblib")
FEATURES = joblib.load("feature_names.joblib")

# Domain mappings
FEATURE_MEANINGS = {
    "LIMIT_BAL": "Credit limit is low relative to usage",
    "MAX_BILL_AMT": "Outstanding bill amount is high",
    "AVG_BILL_AMT": "Average monthly bill is high",
    "TOTAL_PAY_AMT": "Total payments made are low",
    "AVG_PAY_AMT": "Average monthly payment is low",
    "PAY_TO_BILL_RATIO": "Repayment ratio is low",
    "NUM_DELAYED_MONTHS": "History of delayed repayments"
}
NEUTRAL_DEFAULTS = {
    "SEX": 1,                 # Reference group
    "AGE": 40,                # Middle age
    "MARRIAGE": 2,            # Single
    "EDUCATION": 2,           # University
    "AGE_GROUP": 1     # Neutral category
}

def add_neutral_demographics(df):
    for col, val in NEUTRAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = val
    return df

def risk_label(prob):
    if prob >= 0.75:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


EXCLUDED_FROM_EXPLANATION = {
    "SEX", "AGE", "EDUCATION", "MARRIAGE", "AGE_GROUP"
}

def shap_to_text_rules(shap_values, top_k=3, threshold=0.05):
    explanations = []

    shap_dict = dict(zip(FEATURES, shap_values))

    sorted_features = sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for feature, shap_val in sorted_features:
        if feature in EXCLUDED_FROM_EXPLANATION:
            continue
        if abs(shap_val) < threshold:
            continue
        if len(explanations) >= top_k:
            break
        

        direction = "increased" if shap_val > 0 else "reduced"
        meaning = FEATURE_MEANINGS.get(feature, feature)

        explanations.append(
            f"{meaning}, which {direction} the default risk"
        )

    return explanations

def derive_features(df):
    """
    df: DataFrame with raw monthly bill, pay_amt, and pay_status columns
    Returns DataFrame with engineered features exactly as used in training
    """

    bill_cols = [
        "BILL_SEPT", "BILL_AUG", "BILL_JUL",
        "BILL_JUN", "BILL_MAY", "BILL_APR"
    ]

    pay_amt_cols = [
        "PAY_AMT_SEPT", "PAY_AMT_AUG", "PAY_AMT_JUL",
        "PAY_AMT_JUN", "PAY_AMT_MAY", "PAY_AMT_APR"
    ]

    pay_cols = [
        "PAY_SEPT", "PAY_AUG", "PAY_JUL",
        "PAY_JUN", "PAY_MAY", "PAY_APR"
    ]

    # Billing aggregates
    df["MAX_BILL_AMT"] = df[bill_cols].max(axis=1)
    df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)

    # Payment aggregates
    df["TOTAL_PAY_AMT"] = df[pay_amt_cols].sum(axis=1)
    df["AVG_PAY_AMT"] = df[pay_amt_cols].mean(axis=1)

    # EXACT SAME formula as training
    df["PAY_TO_BILL_RATIO"] = (
        df["TOTAL_PAY_AMT"] / ((df[bill_cols].sum(axis=1)) + 1)
    )

    # Repayment behavior
    df["NUM_DELAYED_MONTHS"] = (
        df[pay_cols].apply(lambda x: (x > 0).sum(), axis=1)
    )
    df['NET_BILL'] = df['BILL_SEPT'] - df['BILL_APR']
    df['MAX_DELAY'] = df[pay_cols].max(axis=1)   
    df['AVG_DELAY'] = df[pay_cols].mean(axis=1)
    df['SEVERE_DELAY_FLAG'] = (df[pay_cols] >= 3).any(axis=1).astype(int)     

    return df


def predict_customer(customer_row: pd.DataFrame):
    """
    customer_row: single-row DataFrame with RAW input features
    """

    # Feature engineering
    df = derive_features(customer_row.copy())

    # Inject neutral demographic defaults
    df = add_neutral_demographics(df)

    # Enforce correct feature order
    X = df[FEATURES]

    # Prediction
    prob = model.predict_proba(X)[0][1]
    prediction = int(prob > 0.5)

    # SHAP explanations
    shap_vals = explainer.shap_values(X)[0]

    # Convert SHAP -> text explanations
    rules = shap_to_text_rules(shap_vals, top_k=5, threshold=0.05)

    return {
        "prediction": "Default" if prediction else "No Default",
        "risk_probability": round(prob, 2),
        "risk_level": risk_label(prob),
        "key_reasons": rules
    }




