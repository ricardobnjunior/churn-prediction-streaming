import os
import pandas as pd
from src import data_preprocessing, feature_engineering
from src.predict_churn import load_model, preprocess_input, predict

print("Carregando dados...")
data_path = os.path.join("..\\","data", "processed", "churn_users.csv")
df = data_preprocessing.load_and_clean_data(data_path)

print("Aplicando engenharia de features...")
df_fe = feature_engineering.apply_feature_engineering(df)

# Etapa 3: Salvar dados transformados (opcional)
processed_path = os.path.join("..\\","data", "processed", "churn_users_features.csv")
df_fe.to_csv(processed_path, index=False)
print(f"Dados com features salvos em: {processed_path}")

# Etapa 4: Treinamento do modelo
print("Treinando modelo...")
os.system("python train_model.py")

# Etapa 5: Predição (demonstração com primeiros 10 usuários)
print("Realizando predições de churn...")
model = load_model("../models/churn_model.pkl")
df_demo = predict(preprocess_input(df.head(10)), model)

# Exibir resultados
print("\nResultados de predição:")
print(df_demo[["churn_risk_score", "churn_prediction"]])

