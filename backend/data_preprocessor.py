import pandas as pd
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_METADATA_PATH = PROCESSED_DATA_DIR / "processed_metadata.csv"


def parse_name_list(value):
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item["name"] for item in parsed if isinstance(item, dict) and "name" in item]

def data_preprocessor():
    """
    Функция для предобработки данных фильмов.
    """
    # Загрузка данных
    movies_df = pd.read_csv(RAW_DATA_DIR / "movies_metadata.csv", low_memory=False)
    pp_ratings_df = pd.read_csv(RAW_DATA_DIR / "ratings.csv", low_memory=False)
    keywords_df = pd.read_csv(RAW_DATA_DIR / "keywords.csv", low_memory=False)
    processed_df = movies_df[['id', 'original_title', 'overview', 'adult', 'vote_average']].copy()
    processed_df['id'] = movies_df['id'].astype(str)

    # Форматирование описаний фильмов
    processed_df['overview'] = movies_df['overview'].fillna('').astype(str).str.lower()
    
    # Форматирование стобца adult
    processed_df['adult'] = movies_df['adult'].replace({'True': 1, 'False': 0})
    
    # Создание столбца с жанрами
    processed_df['genres_list'] = movies_df['genres'].apply(parse_name_list)
    
    # Создание бинарного списка animation
    processed_df['animation'] = processed_df['genres_list'].apply(lambda x: 1 if 'Animation' in x else 0)
    
    # Создание столбца ключевых слов
    keywords_df['id'] = keywords_df['id'].astype(str)
    keywords_by_id = keywords_df.drop_duplicates('id').set_index('id')['keywords']
    processed_df['keywords_list'] = processed_df['id'].map(keywords_by_id).apply(parse_name_list)

    # Создание столбца усредненных оценок пользователей
    avg_pp_ratings_df = pp_ratings_df.groupby('movieId', as_index=False)['rating'].mean().reset_index()
    avg_pp_ratings_df.rename(columns={'rating': 'avg_rating'}, inplace=True)

    avg_pp_ratings_df['movieId'] = avg_pp_ratings_df['movieId'].astype(str)

    processed_df['avg_people_rating'] = processed_df['id'].map(avg_pp_ratings_df.set_index('movieId')['avg_rating'])

    #  Создание столбца collection_id
    def extract_collection_id(val):
    # Если значение отсутствует или пустое, вернуть None
      if pd.isnull(val) or val == "":
          return None
      try:
          # Если значение уже является словарём, использовать его напрямую
          if isinstance(val, dict):
              return val.get('id')
          # Если это строка, пытаемся преобразовать в dict
          collection = ast.literal_eval(val)
          if isinstance(collection, dict):
              return collection.get('id')
          return None
      except Exception as e:
          return None
      
    processed_df['collection_id'] = pd.to_numeric(
        movies_df['belongs_to_collection'].apply(extract_collection_id),
        errors='coerce'
    ).astype('Int64')

    # Приводим столбец 'adult' и 'animation' к числовому типу
    processed_df['adult'] = pd.to_numeric(processed_df['adult'], errors='coerce').fillna(0).astype(int)
    processed_df['animation'] = pd.to_numeric(processed_df['animation'], errors='coerce').fillna(0).astype(int)

    # Приводим столбец 'avg_people_rating' и 'vote_average' к floатному типу
    processed_df['avg_people_rating'] = pd.to_numeric(processed_df['avg_people_rating'], errors='coerce')
    processed_df['avg_people_rating'] = processed_df['avg_people_rating'].fillna('No rating')
    processed_df['vote_average'] = pd.to_numeric(movies_df['vote_average'], errors='coerce')
    processed_df['vote_average'] = processed_df['vote_average'].fillna('No rating')

    # Записываем в CSV
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_METADATA_PATH, index=False)

    return True


if __name__ == "__main__":
    data_preprocessor()
