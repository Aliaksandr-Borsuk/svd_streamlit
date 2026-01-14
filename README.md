# "SVD Recommender - Streamlit Demo"

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aliaksandr-borsuk-npp8nu3ljccxffkh8kvf6h.streamlit.app/)

Веб-интерфейс для демонстрации рекомендаций на основе **TruncatedSVD** (разложение матрицы взаимодействий).  
Это продолжение проекта [Recommender_Systems_project](https://github.com/Aliaksandr-Borsuk/Recommender_Systems_project), где был реализован и сравнен ряд классических и нейросетевых алгоритмов. Здесь - минимальный, но рабочий MVP для интерактивного использования .

---

##  Цель
Показать, как модель коллаборативной фильтрации (SVD) может генерировать персонализированные рекомендации фильмов на основе исторических оценок пользователей из датасета **MovieLens 1M**.

---

## Содержание
- `app.py` - Streamlit-интерфейс
- `train_svd.py` - скрипт обучения и сохранения модели
- `svd_model.pkl` - предобученная модель + маппинги (`user_id - index`, `item_id - title`)
- `utils/` - вспомогательные функции для загрузки и предобработки данных
- `requirements.txt` - зависимости для развёртывания

Все данные (**data/**) исключены из репозитория через **.gitignore**.

---

## Локальный запуск

### Требования
- Python 3.10
- Poetry (опционально)

### Установка и запуск
```bash
# Клонировать репозиторий
git clone https://github.com/Aliaksandr-Borsuk/svd_streamlit.git
cd svd_streamlit

# Установить зависимости
poetry install

# Активировать окружение
poetry shell

# Запустить Streamlit
streamlit run app.py 
```

---

Откроется локальный сервер: http://localhost:8501.
  
---

## Онлайн-демо
Развернуто на Streamlit Community Cloud:
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aliaksandr-borsuk-npp8nu3ljccxffkh8kvf6h.streamlit.app/)

---

## Как работает рекомендация?

- Модель обучена на явных рейтингах (rating ∈ {1..5}) из MovieLens.
- Используется sklearn.decomposition.TruncatedSVD для получения:
  - эмбеддингов пользователей (user_factors)
  - эмбеддингов фильмов (item_factors)
- Рекомендации генерируются как скалярное произведение вектора пользователя и всех фильмов → топ-N по скору.
- Поддерживаются только "тёплые" пользователи (те, кто есть в обучающей выборке).

---  
  
## Структура проекта

```  
svd_streamlit/  
├── app.py                  # Streamlit UI  
├── train_svd.py            # Обучение модели  
├── svd_model.pkl           # Сохранённая модель + маппинги  
├── utils/   
│   ├── data_io.py          # Загрузка train/test/meta   
│   └── preprocessing.py    # Построение sparse UI-матрицы   
├── requirements.txt        # Для Streamlit Cloud   
├── .gitignore              # Исключает data/  
└── README.md
```
 
---   
## Связь с основным проектом
Этот репозиторий использует:

- те же данные (warm_train_test_meta), подготовленные в ноутбуке **001_data_and_eda_1m.ipynb**
- ту же логику разбиения и предобработки
результаты экспериментов из **03_01_matrix_factorization_SVD.ipynb**    
Основной проект: [Recommender_Systems_project](https://github.com/Aliaksandr-Borsuk/Recommender_Systems_project)

## Лицензия
MIT License
