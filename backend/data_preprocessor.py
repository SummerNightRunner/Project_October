import pandas as pd
import ast

def data_preprocessor():
    """
    Функция для предобработки данных фильмов.
    """
    # Загрузка данных
    movies_df = pd.read_csv("/Users/summernightrunner/Developer/Project_October/data/movies_metadata.csv", low_memory=False)
    pp_ratings_df = pd.read_csv("/Users/summernightrunner/Developer/Project_October/data/ratings.csv", low_memory=False)
    keywords_df = pd.read_csv("/Users/summernightrunner/Developer/Project_October/data/keywords.csv", low_memory=False)
    
    # Форматирование описаний фильмов
    movies_df['overview'] = movies_df['overview'].fillna('').astype(str).str.lower()
    
    # Форматирование стобца adult
    movies_df['adult'] = movies_df['adult'].replace({'True': 1, 'False': 0})
    
    # Создание столбца с жанрами
    movies_df['genres_list'] = movies_df['genres'].fillna('[]').apply(lambda x: [genre['name'] for genre in ast.literal_eval(x)])
    
    # Создание бинарного списка animation
    movies_df['animation'] = movies_df['genres_list'].apply(lambda x: 1 if 'Animation' in x else 0)
    
    # Создание столбца ключевых слов
    movies_df['keywords_list'] = keywords_df['keywords'].fillna('[]').apply(lambda x: [keyword['name'] for keyword in ast.literal_eval(x)])

    # Создание столбца усредненных оценок пользователей
    avg_pp_ratings_df = pp_ratings_df.groupby('movieId', as_index=False)['rating'].mean().reset_index()
    avg_pp_ratings_df.rename(columns={'rating': 'avg_rating'}, inplace=True)

    movies_df['id'] = movies_df['id'].astype(str)
    avg_pp_ratings_df['movieId'] = avg_pp_ratings_df['movieId'].astype(str)

    movies_df['avg_people_rating'] = movies_df['id'].map(avg_pp_ratings_df.set_index('movieId')['avg_rating'])

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
      
    movies_df['collection_id'] = movies_df['belongs_to_collection'].apply(extract_collection_id)
    movies_df['collection_id'] = movies_df['collection_id'].dropna().astype('Int64')

    # Приводим id к строковому типу
    movies_df['id'] = movies_df['id'].astype(str)

    # Приводим столбец 'adult' и 'animation' к числовому типу
    movies_df['adult'] = pd.to_numeric(movies_df['adult'], errors='coerce').fillna(0).astype(int)
    movies_df['animation'] = pd.to_numeric(movies_df['animation'], errors='coerce').fillna(0).astype(int)

    # Приводим столбец 'avg_people_rating' и 'vote_average' к floатному типу
    movies_df['avg_people_rating'] = movies_df['avg_people_rating'].astype(float)
    if movies_df['avg_people_rating'].isnull().any():
        movies_df['avg_people_rating'] = movies_df['avg_people_rating'].fillna('No rating')
    movies_df['vote_average'] = movies_df['vote_average'].astype(float)
    if movies_df['vote_average'].isnull().any():
        movies_df['vote_average'] = movies_df['vote_average'].fillna('No rating')

    # Записываем в CSV
    movies_df.to_csv('/Users/summernightrunner/Developer/Project_October/data/movies_metadata.csv', index=False)

    return True

data_preprocessor()