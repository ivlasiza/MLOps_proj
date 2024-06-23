import sys
import os
# Получаем правильные пути
SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))      # Каталог со скриптами
PROJECT_PATH = os.path.dirname(SCRIPTS_PATH)                   # Каталог проекта
src_path = os.path.join(PROJECT_PATH, 'src')
sys.path.append(src_path)
print(sys.path)

from fastapi.testclient import TestClient
from api_app_titanic import app


client = TestClient(app)
def test_api_app():
    # Тест 1:  Корректный запрос 
    response = client.post("/predict/",
                           json={"Pclass": 1, "Sex": 0, "Age": 20.0, "SibSp": 0, "Parch": 0, "Fare": 15, "Embarked": 0})
    assert response.status_code == 200
    json_data = response.json()
    assert 'survival_prediction' in json_data

    # Тест 2:  Проверка диапазона предсказания (0 или 1)
    assert json_data['survival_prediction'] in [0, 1]

    # Тест 3:  Обработка отсутствующего признака 
    response = client.post("/predict/",
                           json={"Pclass": 1, "Sex": 0, "SibSp": 0, "Parch": 0, "Fare": 15, "Embarked": 0})
    assert response.status_code == 422  # Должен вернуть ошибку валидации

    # Тест 4:  Некорректный тип данных
    response = client.post("/predict/",
                           json={"Pclass": 1, "Sex": "female", "Age": 20.0, "SibSp": 0, "Parch": 0, "Fare": 15, "Embarked": 0})
    assert response.status_code == 422  # Должен вернуть ошибку валидации
