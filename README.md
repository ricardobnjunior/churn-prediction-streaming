# 🎧 Predição de Churn em Serviços de Streaming com Machine Learning

Este projeto simula um cenário realista de **análise de comportamento de usuários e predição de churn (cancelamento)** em serviços de streaming como **Netflix e Spotify**, utilizando dados sintéticos, engenharia de features comportamentais e modelos de machine learning.

---

## 📌 Objetivos

- Detectar sinais comportamentais que antecedem o churn
- Construir um modelo preditivo robusto usando aprendizado de máquina
- Aplicar segmentações e insights acionáveis para retenção
- Demonstrar o ciclo completo de um projeto de ciência de dados: dados → modelo → ação

---

## 📂 Estrutura do Projeto

```
churn-prediction-streaming/
├── data/
│   ├── raw/                    # Dados originais (ex: streaming_users_behavior.csv)
│   └── processed/              # Dados com features geradas (ex: churn_users.csv)
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict_churn.py
│   └── run_pipeline.py         # Script principal do pipeline
├── models/
│   └── churn_model.pkl         # Modelo treinado com Random Forest
├── reports/
│   └── churn_dashboard_mock.png
├── README.md
└── requirements.txt
```

---

## 🧪 Dataset Simulado

O projeto utiliza um dataset fictício com mais de 5.000 usuários e as seguintes variáveis:

- `user_id`: ID do usuário
- `subscription_type`: tipo de plano (free, premium)
- `total_watch_time_min`: tempo total de consumo
- `num_scroll_events`: número de scrolls sem clicar
- `preferred_genre`: gênero favorito
- `account_age_days`: tempo de conta
- `days_since_last_login`: dias desde último login
- `used_recommendations`: usou sugestões do sistema?
- `num_sessions_last_30d`: sessões nos últimos 30 dias
- `num_genres_watched`: diversidade de consumo
- `churned`: variável alvo

---

## 🚀 Como Executar

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/churn-prediction-streaming.git
cd churn-prediction-streaming
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o pipeline completo:

```bash
python src/run_pipeline.py
```

---

## 📈 Modelo Utilizado

- Algoritmo: `RandomForestClassifier`
- Avaliação: AUC, F1-score, Precision, Recall
- Feature Importance e matriz de confusão disponíveis no notebook

---

## 📊 Exemplos de Visualizações

- Gráfico de importância de variáveis
- Matriz de confusão
- Dashboard com risco de churn por usuário (mock)

---

## 📚 Aprendizados

✅ Como transformar dados brutos em sinais de comportamento  
✅ Engenharia de features aplicadas à retenção  
✅ Predição prática de cancelamento de clientes  
✅ Conexão entre ciência de dados e ações reais de retenção

---

## 🤝 Contribuições

Pull requests são bem-vindos. Para mudanças maiores, por favor abra uma issue antes para discutir o que você gostaria de modificar.

---

## 🧠 Autor

**Ricardo Neves Junior**  
🔗 [LinkedIn](https://www.linkedin.com/in/ricardonevesjunior)

---

## 📝 Licença

MIT © Ricardo Neves Junior
