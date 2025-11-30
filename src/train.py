import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import yaml
import joblib
import os
import mlflow

#название эксперимента
mlflow.set_experiment("Wine Classification")

#параметры
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

#загружаем данные
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')

#признаки и целевая переменная
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

#обучение модели
model = LogisticRegression(
    max_iter=params['train']['max_iter'],
    random_state=params['base']['random_state']
)

#ML flow
with mlflow.start_run():
    print("Starting MLflow run...")

    #лгируем параметры, которые мы использовали
    mlflow.log_param("max_iter", params['train']['max_iter'])
    mlflow.log_param("random_state", params['base']['random_state'])
    mlflow.log_param("model_type", "LogisticRegression")

    #обучаем модель
    model.fit(X_train, y_train)

    #предсказания на тестовой выборке
    predictions = model.predict(X_test)

    #метрика accuracy
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc}")

    #логирование метрики
    mlflow.log_metric("accuracy", acc)

    #сохраняем модель
    model_path = 'model.pkl'
    joblib.dump(model, model_path)

    #логируем модель как артефакт
    mlflow.log_artifact(model_path, artifact_path="model")