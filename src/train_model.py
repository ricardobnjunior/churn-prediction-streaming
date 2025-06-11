import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Caminho do dataset
data_path = os.path.join("..\\","data", "processed", "churn_users.csv")
df = pd.read_csv(data_path)

# Encode de variáveis categóricas
le_subs = LabelEncoder()
le_genre = LabelEncoder()
df["subscription_type"] = le_subs.fit_transform(df["subscription_type"])
df["preferred_genre"] = le_genre.fit_transform(df["preferred_genre"])

# Separar X e y
X = df.drop(columns=["user_id", "churned"])
y = df["churned"]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
print("AUC: {:.4f}".format(roc_auc_score(y_test, y_proba)))

# Salvar modelo
os.makedirs("models", exist_ok=True)
joblib.dump(model, "../models/churn_model.pkl")
print("\n✅ Modelo salvo em models/churn_model.pkl")
