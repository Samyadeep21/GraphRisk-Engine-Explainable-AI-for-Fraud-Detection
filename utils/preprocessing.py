import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ───────────────────────── CORE PREPROCESS FUNCTION
def _preprocess_df(df: pd.DataFrame):
    df = df.copy()

    # ───────── BASIC CLEANING
    df = df.dropna(subset=['nameOrig', 'nameDest', 'amount'])

    # ───────── SAFE COLUMN HANDLING
    if 'type' not in df.columns:
        df['type'] = 'TRANSFER'

    if 'laundering_typology' not in df.columns:
        df['laundering_typology'] = 'normal'

    if 'fraud_probability' not in df.columns:
        df['fraud_probability'] = 0

    if 'isFraud' not in df.columns:
        df['isFraud'] = 0

    if 'isMoneyLaundering' not in df.columns:
        df['isMoneyLaundering'] = 0

    # ───────── ENCODING
    le_type = LabelEncoder()
    df['type_encoded'] = le_type.fit_transform(df['type'])

    le_typology = LabelEncoder()
    df['typology_encoded'] = le_typology.fit_transform(
        df['laundering_typology'].fillna('normal')
    )

    # ───────── SCALING
    scaler = StandardScaler()
    df['amount_scaled'] = scaler.fit_transform(df[['amount']])

    # ───────── TIME FEATURES (FIXED + SAFE)
    if 'step' in df.columns:
        df['hour'] = df['step'] % 24
        df['day_of_week'] = (df['step'] // 24) % 7
    else:
        df['hour'] = 0
        df['day_of_week'] = 0

    # ───────── ADDITIONAL FEATURES (BOOST AUC 🔥)
    df['log_amount'] = np.log1p(df['amount'])
    df['is_large_txn'] = (
        df['amount'] > df['amount'].quantile(0.95)
    ).astype(int)

    # ───────── LABEL
    df['fraud_probability'] = df['fraud_probability'].fillna(0)

    df['is_suspicious'] = (
        (df['isFraud'] == 1) | (df['isMoneyLaundering'] == 1)
    ).astype(int)

    print(f"✅ Loaded {len(df)} transactions")
    print(f"✅ Suspicious transactions: {df['is_suspicious'].sum()}")

    return df


# ───────────────────────── FILE WRAPPER (TRAINING)
def load_and_preprocess(filepath, nrows=None):
    df = pd.read_csv(filepath, nrows=nrows)
    return _preprocess_df(df)


# ───────────────────────── TEST
if __name__ == "__main__":
    df = load_and_preprocess("data/transactions.csv", nrows=100000)
    print(df[['nameOrig', 'nameDest', 'amount_scaled', 'is_suspicious']].head())