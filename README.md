[![LinkedIn – Ricardo Neves Junior](https://img.shields.io/badge/LinkedIn--blue?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ricardonevesjunior)

# 🎧 Churn Prediction in Streaming Services with Machine Learning

This project simulates a realistic scenario of **user behavior analysis and churn prediction** in subscription-based streaming services like **Netflix and Spotify**, using synthetic data, behavioral feature engineering, and machine learning models.

---

## 📌 Objectives

- Detect behavioral signals that precede churn
- Build a robust predictive model using machine learning
- Apply segmentation and actionable insights for user retention
- Demonstrate the complete lifecycle of a data science project: data → model → action

---

## 📂 Project Structure

```
churn-prediction-streaming/
├── data/
│   ├── raw/                    # Raw data (e.g., streaming_users_behavior.csv)
│   └── processed/              # Processed data with features (e.g., churn_users.csv)
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict_churn.py
│   └── run_pipeline.py         # Main pipeline script
├── models/
│   └── churn_model.pkl         # Trained Random Forest model
├── reports/
│   └── churn_dashboard_mock.png
├── README.md
└── requirements.txt
```

---

## 🧪 Simulated Dataset

The project uses a synthetic dataset with over 5,000 users and the following features:

- `user_id`: user ID
- `subscription_type`: subscription plan (free, premium)
- `total_watch_time_min`: total content consumption time
- `num_scroll_events`: number of scrolls without clicking
- `preferred_genre`: favorite genre
- `account_age_days`: account age
- `days_since_last_login`: days since last login
- `used_recommendations`: whether recommendations were used
- `num_sessions_last_30d`: sessions in the last 30 days
- `num_genres_watched`: content diversity
- `churned`: target variable

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/churn-prediction-streaming.git
cd churn-prediction-streaming
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the full pipeline:

```bash
python src/run_pipeline.py
```

---

## 📈 Model Used

- Algorithm: `RandomForestClassifier`
- Evaluation metrics: AUC, F1-score, Precision, Recall
- Feature importance and confusion matrix available in the notebook

---

## 📊 Visualization Examples

- Feature importance bar chart
- Confusion matrix
- Churn risk dashboard per user (mock)

---

## 📚 Key Learnings

✅ How to turn raw data into behavioral signals  
✅ Feature engineering applied to retention modeling  
✅ Practical churn prediction in real use cases  
✅ Bridging data science with real-world business actions

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 🧠 Author

**Ricardo Neves Junior**  
🔗 [LinkedIn](https://www.linkedin.com/in/ricardo-neves-junior/)

---

## 📝 License

MIT © Ricardo Neves Junior
