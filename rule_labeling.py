import pandas as pd

def rule_based_label(row):
    if row['Debt/Equity'] is not None and row['Debt/Equity'] > 3.5:
        return "Risky"
    elif row['ROE'] is not None and row['ROE'] > 0.15 and row['Operating Margin'] > 0.15:
        return "Safe"
    else:
        return "Medium"

def classify_price_risk(row):
    if row['price_volatility'] < 0.015 and row['12m_return'] > 0.10 and row['max_drawdown'] > -0.20:
        return "Stable"
    elif row['price_volatility'] < 0.03:
        return "Moderate"
    else:
        return "Speculative"

# Apply Rule-based Labeling to financial dataset
try:
    df = pd.read_csv("data/processed/financial_statements_with_z.csv")
    df['Rule_Label'] = df.apply(rule_based_label, axis=1)
    df.to_csv("data/processed/financial_statements_with_z_and_rule.csv", index=False)
    print("✅ Rule-based labels added.")
except FileNotFoundError:
    print("❌ File not found: financial_statements_with_z.csv")

# Apply Price Risk Classification
try:
    df = pd.read_csv("data/processed/financial_with_price.csv")
    df['Price_Risk_Category'] = df.apply(classify_price_risk, axis=1)
    df.to_csv("data/processed/financial_with_price_labeled.csv", index=False)
    print("✅ Price risk categories added.")
except FileNotFoundError:
    print("❌ File not found: financial_with_price.csv")