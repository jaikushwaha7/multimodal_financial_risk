import pandas as pd

def compute_altman_z(row):
    try:
        T1 = row['Working Capital'] / row['Total Assets']
        T2 = row['Retained Earnings'] / row['Total Assets']
        T3 = row['EBIT'] / row['Total Assets']
        T4 = row['Market Cap'] / row['Total Liabilities']
        T5 = row['Total Revenue'] / row['Total Assets']
        z = 1.2*T1 + 1.4*T2 + 3.3*T3 + 0.6*T4 + 1.0*T5
        return z
    except:
        return None

def label_from_z(z):
    if z is None:
        return "Unknown"
    elif z > 2.99:
        return "Safe"
    elif z >= 1.8:
        return "Medium"
    else:
        return "Risky"

try:
    df = pd.read_csv("data/raw/financial_statements.csv")
    df["Altman_Z"] = df.apply(compute_altman_z, axis=1)
    df["Z_Label"] = df["Altman_Z"].apply(label_from_z)
    df.to_csv("data/processed/financial_statements_with_z.csv", index=False)
    print("✅ Z-scores and labels saved.")
except Exception as e:
    print(f"❌ Error computing Z-scores: {e}")
