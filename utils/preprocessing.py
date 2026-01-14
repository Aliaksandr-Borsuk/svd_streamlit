# импорты
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def prepare_knn_matrix(
    df: pd.DataFrame,
    threshold: float = 0.0
) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
    """
    Преобразует df в sparse матрицу взаимодействий для KNN-моделей на основе implicit feedback.

    Args:
        df (pd.DataFrame): Данные с колонками ['user_id', 'item_id', 'rating']
        threshold (float): Минимальный рейтинг для учета взаимодействия (по умолчанию 0.0)

    Returns:
        Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
            - interaction_matrix: csr_matrix (users × items)
            - user_to_index: словарь user_id → индекс
            - item_to_index: словарь item_id → индекс
    """
    # Фильтрация по порогу
    implicit_df = df[df['rating'] > threshold].copy()

    # Построение словарей индексации
    user_to_index = {user_id: idx for idx, user_id in enumerate(implicit_df['user_id'].unique())}
    item_to_index = {item_id: idx for idx, item_id in enumerate(implicit_df['item_id'].unique())}

    # ID в индексы
    row_indices = implicit_df['user_id'].map(user_to_index).values
    col_indices = implicit_df['item_id'].map(item_to_index).values

    # Все взаимодействия — единицы
    data = [1] * len(implicit_df)

    # sparse матрица
    interaction_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(user_to_index), len(item_to_index))
    )

    return interaction_matrix, user_to_index, item_to_index



def prepare_ui_matrix(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'item_id',
    rating_col: str = 'rating',
    implicit: bool = True,
    threshold: Optional[float] = 0.0,   # для implicit: rating > threshold = 1
    center: Optional[str] = None,       # None | 'user' | 'item' | 'both' (только для explicit)
    normalize: Optional[str] = None,    # None | 'zscore' | 'minmax' (только для explicit)
    dtype: np.dtype = np.float32
) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
    """
    ###         расширенная версия prepare_knn_matrix          ###

    Преобразует df в sparse матрицу взаимодействий users * items.
    - implicit-True: бинаризация по threshold (rating > threshold -> 1),
      без центрирования/нормализации.
    - implicit=False: explicit рейтинги с опциональными center и normalize.
    Args:
        df (pd.DataFrame): Данные с колонками ['user_id', 'item_id', 'rating']
        threshold (float): Минимальный рейтинг для учета взаимодействия (по умолчанию 0.0)

    Returns:
        Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
            - interaction_matrix: csr_matrix (users × items)
            - user_to_index: словарь user_id → индекс
            - item_to_index: словарь item_id → индекс
    """
    # Базовая выборка нужных столбцов
    df = df[[user_col, item_col, rating_col]].copy()

    # Фильтрация по порогу
    # implicit or explicit
    if implicit:
        if threshold is not None:
            df = df[df[rating_col] > threshold]
        # Всё что осталось = 1.0
        df[rating_col] = 1.0
    else:
        # explicit: без фильтрации (или можно использовать threshold как минимальный рейтинг)
        if threshold is not None:
            df = df[df[rating_col] > threshold]
        # convert to float32 для совместимости с матричными операциями
        df[rating_col] = df[rating_col].astype(np.float32)

    # Построение словарей индексации
    user_to_index = {user_id: idx for idx, user_id in enumerate(df[user_col].unique())}
    item_to_index = {item_id: idx for idx, item_id in enumerate(df[item_col].unique())}

    # ID в индексы
    row_indices = df[user_col].map(user_to_index).values
    col_indices = df[item_col].map(item_to_index).values

    # Все взаимодействия
    data = df[rating_col].astype(dtype).values

    # мастерим sparse матрицу
    X = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(user_to_index), len(item_to_index)),
        dtype=dtype
    )

    # Центрирование и нормализация только для explicit
    if not implicit:
        X = X.tocsr() # на всякий...

        # Центрирование
        if center in ('user', 'both'):
            # вычитаем средний рейтинг пользователя из его ненулевых
            sums = X.sum(axis=1).A1
            counts = np.diff(X.indptr)
            # Среднее рейтингов для каждого пользователя;
            # для пользователей без ненулевых значений оставляем 0.
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            # центрируем по среднему из всех ненулевых
            row_indices = np.repeat(np.arange(X.shape[0]), counts)
            X.data -= means[row_indices]

        if center in ('item', 'both'):
            # вычитаем средний рейтинг айтема из его ненулевых
            Xc = X.tocsc()
            sums = Xc.sum(axis=0).A1
            item_counts = np.diff(Xc.indptr)
            means = np.divide(sums, item_counts, out=np.zeros_like(sums), where=item_counts > 0)
            col_indices = np.repeat(np.arange(Xc.shape[1]), item_counts)
            Xc.data -= means[col_indices]
            X = Xc.tocsr()

        # Нормализация
        if normalize == 'zscore':
            # для каждого пользователя считаем std через группировку
            counts = np.diff(X.indptr)              # количество ненулевых элементов в каждой строке
            sums = X.sum(axis=1).A1                 # сумма значений по строке
            sq_sums = X.power(2).sum(axis=1).A1     # сумма квадратов значений по строке
            means = sums / counts                   # среднее по строке
            stds = np.sqrt(sq_sums / counts - means**2)    # стандартное отклонени
            # индексы строк для каждого элемента data
            row_indices = np.repeat(np.arange(X.shape[0]), counts)

            # нормализация только там, где std > 0
            valid = stds[row_indices] > 0
            X.data[valid] /= stds[row_indices][valid]

        elif normalize == 'minmax':
            # по пользователю: (x - min) / (max - min)
            for u in range(X.shape[0]):
                s, e = X.indptr[u], X.indptr[u + 1]
                if e > s:
                    segment = X.data[s:e]
                    mn, mx = segment.min(), segment.max()
                    rng = mx - mn
                    if rng > 0:
                        X.data[s:e] = (segment - mn) / rng
                    else:
                        # если все одинаковые - оставляем как есть
                        pass

    return X, user_to_index, item_to_index
