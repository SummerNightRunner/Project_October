from recommendations_func import get_recommendations
from user_registration import register_user
from ratings_updater import ratings_updater
import pandas as pd

# Пример работы
username = input("Юзернейм: ")
email = input("Почта: ")
password = input("Пароль: ")
while True:
  confirm_password = input("Повторите пароль: ")
  if confirm_password == password:
    break
  print("Несовпадающие пароли, повторите.")

register_user(email, username, password)

movie_names = []
print("Введите названия фильмов (введите 'q' для завершения):")
while True:
  movie_name = input()
  if movie_name.lower() == 'q':
    break
  movie_names.append(movie_name)

movies_df = pd.read_csv('data/movies_metadata.csv', low_memory=False)
movie_ids = []
for movie_name in movie_names:
  matching_movies = movies_df[movies_df['title'].str.lower() == movie_name.lower()]
  if not matching_movies.empty:
    movie_ids.append(matching_movies['id'].values[0])
  else:
    print(f"Фильм '{movie_name}' не найден в базе данных.")

recomendation = get_recommendations(movie_ids)
print(recomendation)
