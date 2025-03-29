import csv
import os
from cryptography.fernet import Fernet

# Генерация или загрузка ключа шифрования
KEY_FILE = '../data/secret.key'

def load_or_generate_key():
  if not os.path.exists(KEY_FILE):
    key = Fernet.generate_key()
    with open(KEY_FILE, 'wb') as key_file:
      key_file.write(key)
  else:
    with open(KEY_FILE, 'rb') as key_file:
      key = key_file.read()
  return key

key = load_or_generate_key()
cipher = Fernet(key)

# Путь к базе данных
DATABASE_FILE = '../data/users.csv'

# Инициализация базы данных, если она не существует
if not os.path.exists(DATABASE_FILE):
  with open(DATABASE_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'email', 'username', 'password'])  # Заголовки колонок

# Функция для получения следующего ID
def get_next_user_id():
  try:
    with open(DATABASE_FILE, 'r') as file:
      reader = csv.reader(file)
      next(reader)  # Пропускаем заголовок
      rows = list(reader)
      if rows:
        last_id = int(rows[-1][0])
        return last_id + 1
      else:
        return 270897  # ID первого пользователя
  except FileNotFoundError:
    return 270897

# Функция для регистрации пользователя
def register_user(email, username, password):
  user_id = get_next_user_id()
  encrypted_email = cipher.encrypt(email.encode()).decode()
  encrypted_username = cipher.encrypt(username.encode()).decode()
  encrypted_password = cipher.encrypt(password.encode()).decode()

  with open(DATABASE_FILE, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([user_id, encrypted_email, encrypted_username, encrypted_password])

  return user_id

# Пример использования
if __name__ == '__main__':
  print("Регистрация нового пользователя")
  email = input("Введите email: ")
  username = input("Введите логин: ")
  password = input("Введите пароль: ")

  user_id = register_user(email, username, password)
  print(f"Пользователь успешно зарегистрирован с ID: {user_id}")