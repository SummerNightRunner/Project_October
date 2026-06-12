from pathlib import Path

import pandas as pd
from data_preprocessor import data_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MOVIES_METADATA_PATH = RAW_DATA_DIR / "movies_metadata.csv"
RATINGS_PATH = RAW_DATA_DIR / "ratings.csv"


def ratings_updater(movie_name, user_rating, user_id):
    """
    Добавляет новый рейтинг от пользователя в файл ratings.csv
    Обновляет файл avg_ratings.csv, вычисляя новый средний рейтинг для выбранного фильма.
    """
    # Загрузка данных для поиска id фильма
    movies_df = pd.read_csv(MOVIES_METADATA_PATH, low_memory=False)
    movie_id = movies_df[movies_df['title'].str.lower() == movie_name.lower()]['id'].values[0]
    
    # Создание DataFrame для новой строки
    new_row_df = pd.DataFrame({
        'userId': [user_id],
        'movieId': [movie_id],
        'rating': [float(user_rating)],
        'timestamp': ['']
    })

    # Добавление новой строки в CSV-файл в режиме дозаписи
    new_row_df.to_csv(RATINGS_PATH, mode='a', header=False, index=False)

    # Обновление информации о фильмах (убедитесь, что эта функция не дублирует данные)
    data_preprocessor()

    return True
