import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Загрузка модели
with open('svd_model.pkl', 'rb') as f:
    data = pickle.load(f)

user_factors = data['user_factors']
item_factors = data['item_factors']
user2index = data['user2index']
index2item = data['index2item']

# загрузка названий
item_id_to_title = data.get('item_id_to_title', None)

# Список известных пользователей
known_users = sorted(user2index.keys())
min_uid, max_uid = min(known_users), max(known_users)
example_users = ", ".join(map(str, known_users[:5]))


# Интерфейс
st.title("SVD Рекомендации")
st.caption(f"Для обучения модели использовались обезличеные данные.\
            \nПоэтому у нас нет имён , есть только user_id ")
st.caption(f"Известные user_id: от {min_uid} до {max_uid}. Примеры: {example_users}")

user_id = st.number_input("User ID", min_value=1, value=min_uid)

# Проверка
if user_id not in user2index:
    st.error(f"Пользователь {user_id} не найден в обучающих данных.")
    st.stop()  # останавливает выполнение дальше


n_rec = st.slider("Количество рекомендаций", 1, 20, 5)

if st.button("Получить"):
    u_idx = user2index[user_id]
    scores = user_factors[u_idx] @ item_factors.T
    top_indices = np.argsort(-scores)[:n_rec]
    rec_item_ids = [index2item[i] for i in top_indices]

    st.write("Рекомендации:")
    for item_id in rec_item_ids:
        st.write(f"- {item_id_to_title.get(item_id, item_id)}")
