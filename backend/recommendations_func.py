import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Загрузка данных
movies_df = pd.read_csv("/Users/summernightrunner/Developer/Project_October/data/movies_metadata.csv", low_memory=False)

# Форматирование описаний фильмов
movies_df['overview'] = movies_df['overview'].fillna('')

# Векторизация описаний фильмов
vectorizer = TfidfVectorizer(lowercase=True, max_features=1000, min_df=10, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(movies_df['overview'])

# Векторизация жанров
mlb = MultiLabelBinarizer()
genres_matrix = mlb.fit_transform(movies_df['genres_list'])

# Массив флагов
flags_array = np.array(movies_df[['animation', 'adult']].values)

# Векторизация ключевых слов
mlb_keywords = MultiLabelBinarizer()
keywords_matrix = mlb_keywords.fit_transform(movies_df['keywords_list'].fillna('[]'))

# схожесть по бинарным признакам
def similarity_flags(profile_flags, flags_array):
    """
    Вычисляет схожесть по бинарным признакам.
    Для каждого фильма считает долю совпадающих флагов с профилем пользователя.
    
    profile_flags: вектор выбранных фильмов (усреднённый), например, [is_animation, is_adult]
    flags_array: матрица бинарных флагов для всех фильмов, shape (n_samples, n_flags)
    
    Возвращает массив схожести, где 1.0 – все признаки совпадают, 0.0 – ни один.
    """
    # Приводим профиль к одномерному массиву
    profile_flags = np.asarray(profile_flags).flatten()
    # Считаем число совпадений для каждой строки
    matches = (flags_array == profile_flags).sum(axis=1)
    # Делим на общее количество признаков, чтобы получить долю совпадений
    return matches / len(profile_flags)

# Функция для получения рекомендаций
def get_recommendations(selected_movie_ids, include_adult=False, top_n=20,
                                      weight_desc=0.35, weight_collection=0.1, weight_genres=0.15, weight_flags=0.2, weight_keywords=0.2):
    """
    Расширенная функция рекомендаций, учитывающая:
      - Описание фильма (TF‑IDF векторизация overview)
      - Жанры фильма (бинарное представление жанров)
      - Бинарные флаги (например, is_animation, is_adult)
      - Ключевые слова (полученные из keywords.csv)
    
    Аргументы:
      selected_movie_ids: список ID фильмов, выбранных пользователем.
      include_adult: если False, фильмы для взрослых исключаются.
      top_n: количество рекомендаций.
      weight_desc, weight_genres, weight_flags, weight_keywords: веса для каждого компонента.
      
    Требуется, чтобы в глобальной области видимости были определены:
      - movies_df (DataFrame с информацией о фильмах)
      - tfidf_matrix (матрица TF‑IDF описаний)
      - genres_matrix (матрица жанров)
      - flags_array (массив бинарных признаков)
      - keywords_matrix (матрица ключевых слов)
    """
    # Фильтрация выбранных фильмов
    mask_selected = movies_df['id'].isin(selected_movie_ids).values
    selected_vectors = tfidf_matrix[mask_selected]
    selected_genres = genres_matrix[mask_selected]
    selected_flags = flags_array[mask_selected]
    selected_keywords = keywords_matrix[mask_selected]
    
    if selected_vectors.shape[0] == 0:
        raise ValueError("No valid movie IDs provided.")
    
    # Формируем профиль пользователя как среднее выбранных фильмов
    profile_desc = np.asarray(selected_vectors.mean(axis=0))
    profile_genres = selected_genres.mean(axis=0)
    profile_flags = selected_flags.mean(axis=0)
    profile_keywords = selected_keywords.mean(axis=0)
    
    # Фильтрация датасета по предпочтению взрослых фильмов
    if not include_adult:
        mask_adult = (movies_df['adult'] == 0).values
        tfidf_matrix_filtered = tfidf_matrix[mask_adult]
        genres_matrix_filtered = genres_matrix[mask_adult]
        flags_array_filtered = flags_array[mask_adult]
        keywords_matrix_filtered = keywords_matrix[mask_adult]
        movies_df_filtered = movies_df[mask_adult].reset_index(drop=True)
    else:
        tfidf_matrix_filtered = tfidf_matrix
        genres_matrix_filtered = genres_matrix
        flags_array_filtered = flags_array
        keywords_matrix_filtered = keywords_matrix
        movies_df_filtered = movies_df.copy().reset_index(drop=True)
    
    # Вычисляем схожесть для каждого источника признаков:
    sim_desc = cosine_similarity(profile_desc, tfidf_matrix_filtered).flatten()
    sim_genres = cosine_similarity(profile_genres.reshape(1, -1), genres_matrix_filtered).flatten()
    sim_keywords = cosine_similarity(profile_keywords.reshape(1, -1), keywords_matrix_filtered).flatten()
    sim_flags = similarity_flags(profile_flags, flags_array_filtered)

    # Определяем бонус по коллекциям: Находим коллекции выбранных фильмов (если фильм принадлежит коллекции, там хранится id коллекции)
    selected_collections = set(movies_df[movies_df['id'].isin(selected_movie_ids)]['collection_id'].dropna())
    # Для каждого фильма из отфильтрованного датасета проверяем, принадлежит ли он к одной из этих коллекций
    sim_collection = np.array([
        1.0 if row['collection_id'] in selected_collections else 0.0
        for idx, row in movies_df_filtered.iterrows()
    ])
    
    # Комбинируем схожести с заданными весами
    combined_sim = (weight_desc * sim_desc +
                    weight_genres * sim_genres +
                    weight_flags * sim_flags +
                    weight_keywords * sim_keywords +
                    weight_collection * sim_collection)
    
    # Сортировка по убыванию комбинированного сходства
    sorted_indices = np.argsort(combined_sim)[::-1]
    
    # Собираем первые top_n фильмов по схожести
    recommendations = []
    for idx in sorted_indices:
        movie_id = movies_df_filtered.iloc[idx]['id']
        if movie_id in selected_movie_ids:
            continue
        recommendations.append({
            'title': movies_df_filtered.iloc[idx]['title'],
            'vote_average': movies_df_filtered.iloc[idx]['vote_average'],
            'site_user_rating': movies_df_filtered.iloc[idx]['avg_people_rating'],
        })
        if len(recommendations) == top_n:
            break
    
    # Сортируем массив топ_n рекомендаций по рейтингу (от высокого к низкому)
    recommendations = sorted(recommendations, key=lambda x: x['vote_average'], reverse=True)
    
    return recommendations

selected_movie_ids = ['862']
recommendations = get_recommendations(selected_movie_ids)
print(recommendations)