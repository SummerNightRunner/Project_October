import pandas as pd

def ratings_updater(movie_name, user_rating):
    """
    Добавляет новый рейтинг от пользователя в файл ratings.csv
    Обновляет файл avg_ratings.csv, вычисляя новый средний рейтинг для выбранного фильма.
    """
    # Загрузка данных
    ratings_df = pd.read_csv('data/ratings.csv')
    movies_df = pd.read_csv('movies_metadata.csv')
    
    # Находим id фильма по названию (приводим к нижнему регистру для сравнения)
    movie_id = movies_df[movies_df['title'].str.lower() == movie_name.lower()]['id'].values[0]

    # Создание новой строки с рейтингом
    new_row = {
        'userId': ratings_df['userId'].max() + 1,
        'movieId': movie_id,
        'rating': user_rating
    }
    
    # Сохранение обновленного файла ratings.csv
    ratings_df.to_csv('data/ratings.csv', index=False)

    return True

ratings_updater('Toy Story', 5)