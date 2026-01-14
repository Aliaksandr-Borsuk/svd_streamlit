# импорты
from pathlib import Path
import json
import pandas as pd
from typing import Tuple, Dict

def train_test_reader(d_path: Path | str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Читает train.csv, test.csv и meta.json в папке d_path.
    Возвращает (train_df, test_df, meta_dict).
    """
    d_path = Path(d_path)
    if not d_path.exists():
        raise FileNotFoundError(f"Путь к данным не найден: {d_path}")
    meta_file = d_path / 'meta.json'
    train_file = d_path / 'train.csv'
    test_file  = d_path / 'test.csv'
    missing = [p.name for p in (meta_file, train_file, test_file) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Отсутствуют файлы {d_path}: {', '.join(missing)}")
    with meta_file.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    train = pd.read_csv(train_file)
    test  = pd.read_csv(test_file)
    return train, test, meta
