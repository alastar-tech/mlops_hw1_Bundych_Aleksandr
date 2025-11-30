import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os


#cоздаем папки для обработанных данных
os.makedirs('data/processed', exist_ok=True)

#загружаем параметры
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

#загружаем исходный датасет
df = pd.read_csv('data/raw/wine.csv')

#сплитим данные
train, test = train_test_split(
    df,
    test_size=params['prepare']['split_ratio'],
    random_state=params['base']['random_state']
)

#сохраняем обучающую и тестовую выборки
train.to_csv('data/processed/train.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)