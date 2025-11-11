import os
import joblib

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import mlflow
import mlflow.sklearn

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity


from evidently.report import Report
from evidently.metric_preset import DataQualityPreset, DataDriftPreset

import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Загрузка и подготовка данных"""
    # Далеко ходить не будем а возьмем банальные ирисы
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Сохраняем данные для послежующего использования
    os.makedirs("data/", exist_ok=True)
    df.to_csv('data/iris.csv', index=False)
    return df



def deepchecks_analysis(df):
    """Анализ данных с помощью Deepchecks"""
    dataset = Dataset(df, label='target', cat_features=[])
    integrity_suite = data_integrity()
    integrity_result = integrity_suite.run(dataset)
    
    # Сохраняем отчет
    os.makedirs("reports/", exist_ok=True)
    integrity_result.save_as_html('reports/deepchecks_report.html')
    print("Deepchecks отчет сохранен в reports/deepchecks_report.html")
    
    return integrity_result



def evidently_analysis(df):
    """Анализ дрейфа данных с EvidentlyAI"""
    
    # Разделяем данные на референсные и текущие
    reference_data = df.sample(frac=0.7, random_state=42)
    current_data = df.drop(reference_data.index)
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Сохраняем отчет
    os.makedirs("reports/", exist_ok=True)
    report.save_html('reports/evidently_report.html')
    print("EvidentlyAI отчет сохранен в reports/evidently_report.html")
    
    return report


def mlflow_experiment(df):
    """Эксперимент с MLflow"""
    
    # Подготовка данных
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Настройка MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Iris_Classification")
    
    with mlflow.start_run():
        # Параметры модели
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        # Обучение модели
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Предсказания и метрики
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Подпись модели
        input_example = X_test.iloc[:3] 
        
        # Логирование в MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            model, 
            name="random_forest_model",
            input_example=input_example
        )
        
        # Сохранение модели
        os.makedirs("models/", exist_ok=True)
        joblib.dump(model, 'models/iris_model.pkl')
        print(f"Модель обучена. Accuracy: {accuracy:.4f}")
        print("Модель и метрики залогированы в MLflow с input_example и signature.")
        
        return model, accuracy

def main():
    """Основной пайплайн"""
    print("Запуск ML пайплайна...")
    
    # 1. Загрузка данных
    df = load_and_prepare_data()
    print(f"Данные загружены. Размер: {df.shape}")
    
    # 2. Deepchecks анализ
    deepchecks_analysis(df)
    
    # 3. EvidentlyAI анализ дрейфа
    evidently_analysis(df)
    
    # 4. MLflow эксперимент
    model, accuracy = mlflow_experiment(df)
    print("Результаты:")
    print("- Deepchecks отчет: reports/deepchecks_report.html")
    print("- EvidentlyAI отчет: reports/evidently_report.html") 
    print("- MLflow эксперименты: ./mlruns")
    print("- Модель: models/iris_model.pkl")

if __name__ == "__main__":
    main()