# ML Pipeline with CI/CD

Домашнее задание по модулю «CI/CD для ML».  
Реализован воспроизводимый ML-пайплайн с интеграцией GitHub Actions и GitLab CI.

## Возможности
- Загрузка и сохранение данных (Iris)
- Проверка качества данных с **Deepchecks**
- Анализ дрейфа признаков с **EvidentlyAI**
- Обучение модели и логирование эксперимента в **MLflow**
- Автоматическая сборка и артефакты в CI

## После запуска:

Отчёты: reports/deepchecks_report.html, reports/evidently_report.html
Модель: models/iris_model.pkl

## CI/CD
GitHub Actions: запускается при пуше в main
GitLab CI: запускается при пуше в main
Артефакты (отчёты, модель) сохраняются в каждом запуске