import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

def load_model(path='models/churn_model.pkl'):
    return joblib.load(path)

def preprocess_input(df):
    df = df.copy()

    # Remove colunas que não são features
    if "user_id" in df.columns:
        df = df.drop(columns=["user_id"])
    if "churned" in df.columns:
        df = df.drop(columns=["churned"])

    # Codificação
    le_subs = LabelEncoder()
    le_genre = LabelEncoder()
    df["subscription_type"] = le_subs.fit_transform(df["subscription_type"])
    df["preferred_genre"] = le_genre.fit_transform(df["preferred_genre"])

    # Features comportamentais
    df["watch_time_per_session"] = df["total_watch_time_min"] / (df["num_sessions_last_30d"] + 1e-5)
    df["scrolls_per_session"] = df["num_scroll_events"] / (df["num_sessions_last_30d"] + 1e-5)
    df["avg_days_between_logins"] = df["account_age_days"] / (df["num_sessions_last_30d"] + 1e-5)
    df["genre_diversity_index"] = df["num_genres_watched"] / (df["account_age_days"] + 1)
    df["is_engaged"] = (
        (df["total_watch_time_min"] > 400) &
        (df["used_recommendations"] == True) &
        (df["days_since_last_login"] < 7)
    ).astype(int)

    return df

def predict(df, model):
    X = df.copy()
    df["churn_risk_score"] = model.predict_proba(X)[:, 1]
    df["churn_prediction"] = (df["churn_risk_score"] > 0.5).astype(int)
    return df

if __name__ == "__main__":
    # Exemplo de uso
    path = os.path.join("data", "processed", "churn_users.csv")
    df = pd.read_csv(path).head(10)
    model = load_model()
    df_result = predict(preprocess_input(df), model)
    print(df_result[["churn_risk_score", "churn_prediction"]])
