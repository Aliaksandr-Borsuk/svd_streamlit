import pandas as pd
import numpy as np
import random
import pickle
from pprint import pprint
from pathlib import Path
from sklearn.decomposition import TruncatedSVD

from utils.data_io import train_test_reader # для чтения сохранённых
from utils.preprocessing import prepare_ui_matrix  # для получения матрицы взаимодействий

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

train_tast_path = Path('data/warm_train_test_meta')

train, test, meta = train_test_reader(train_tast_path)
pprint(meta, width=80, compact=False)
print(f'\ntrain shape : {train.shape}')
print(f'test shape  : {test.shape}')
# print( '\n', '*'*50, '\ntrain.head')
# print(train.head(3))
# print('\n', '*'*50, '\ntest.head')
# print(test.head(3))

# получаем матрицу взаимодействий и словари
# Лучшие параметры truncated:
# {'n_components': 5, 'n_iter': 42, 'center': None, 'normalize': None}
explicit_train_matrix, user2index, item2index =\
    prepare_ui_matrix(
                    train,
                    user_col='user_id',
                    item_col='item_id',
                    rating_col='rating',
                    implicit=False,       # работаем с explicit рейтингами
                    threshold=None,       # не фильтруем по порогу что бы матрица
                                        # была соразмерна implisit матрице
                    center= None,       # подбираем
                    normalize=None   # подбираем
                )
print('***  matrix is done ' )

# создаём модель
svd = TruncatedSVD(n_components=5, random_state=RANDOM_STATE, n_iter=42 )
svd.fit(explicit_train_matrix)

# Сохраняем модель и маппинги, user_factors, item_factors
user_factors = svd.transform(explicit_train_matrix)  # shape: (n_users, n_components)
item_factors = svd.components_.T                     # shape: (n_items, n_components)

# Собираем маппинг item_id → title
item_id_to_title = train.drop_duplicates('item_id').set_index('item_id')['title'].to_dict()

with open('svd_model.pkl', 'wb') as f:
    pickle.dump({
        'user_factors': user_factors,
        'item_factors': item_factors,
        'user2index': user2index,
        #'index2user': {v: k for k, v in user2index.items()},
        #'item2index': item2index,
        'index2item': {v: k for k, v in item2index.items()},
        'item_id_to_title': item_id_to_title
    }, f)

print("The model and some things were saved in the svd_model.pkl")