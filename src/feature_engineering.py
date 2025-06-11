import pandas as pd
from sklearn.preprocessing import LabelEncoder

def apply_feature_engineering(df):
    le_subs = LabelEncoder()
    le_genre = LabelEncoder()

    df["subscription_type"] = le_subs.fit_transform(df["subscription_type"])
    df["preferred_genre"] = le_genre.fit_transform(df["preferred_genre"])

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
