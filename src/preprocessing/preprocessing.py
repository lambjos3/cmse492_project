import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and removal of unusable columns."""
    drop_cols = ["SEED", "POSTSEASON"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced ML features."""
    df = df.copy()
    df["eff_margin"] = df["ADJOE"] - df["ADJDE"]
    df["tov_control"] = (1 - df["TOR"]) + df["TORD"]
    df["shooting_eff"] = df["EFG_O"] - df["EFG_D"]
    df["tempo_adj_rating"] = df["eff_margin"] * df["ADJ_T"]
    return df

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode tournament appearance (1/0)."""
    df["tourney_target"] = (df["POSTSEASON"].notna() & (df["POSTSEASON"] != "NA")).astype(int)
    return df

def split_by_year(df: pd.DataFrame, train_until=2022):
    """Temporal split to prevent data leakage."""
    train = df[df["YEAR"] <= train_until]
    test = df[df["YEAR"] > train_until]
    return train, test

def scale_features(train_X, test_X):
    """Scale numeric features for linear/boosting models."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X)
    test_scaled = scaler.transform(test_X)
    return train_scaled, test_scaled, scaler
