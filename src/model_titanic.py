import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle

# Получаем правильные пути
SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))      # Каталог со скриптами
PROJECT_PATH = os.path.dirname(SCRIPTS_PATH)                   # Каталог проекта
DATASETS_PATH = os.path.join(PROJECT_PATH, "datasets")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")

# Создаем директорию для моделей, если ее нет
os.makedirs(MODELS_PATH, exist_ok=True)

# Загрузка датасета Titanic из файла CSV
train_df = pd.read_csv(os.path.join(DATASETS_PATH, 'dataset_titanic.csv'))

# Предобработка данных
def preprocess_data(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    label_encoders = {}
    for col in ['Sex', 'Embarked']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return df

train_df = preprocess_data(train_df.copy())

# Разделение данных на признаки и целевую переменную
X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']

# Масштабирование числовых признаков
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение гиперпараметров для Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Обучение модели случайного леса с Grid Search
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Лучшая модель
best_rf_model = grid_search.best_estimator_

# Предсказание на тестовом наборе
y_pred = best_rf_model.predict(X_test)
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]  # Вероятности для ROC AUC

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

roc_auc = roc_auc_score(y_test, y_pred_proba)  # Используем вероятности
print(f"ROC AUC: {roc_auc}")

print('Classification Report:')
print(classification_report(y_test, y_pred))

# Сохраняем модель
model_path = os.path.join(MODELS_PATH, 'model_titanic.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(best_rf_model, file)
    print(f'The model was saved successfully: {model_path}')
