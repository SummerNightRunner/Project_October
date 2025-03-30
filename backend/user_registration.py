import os
import pandas as pd
import hashlib
from cryptography.fernet import Fernet
from re import match

# Пути к файлам
KEY_FILE = 'data/secret.key'
DATABASE_FILE = 'data/users.csv'

# Определяем имена столбцов для базы данных
COLUMNS = ['id', 'email', 'email_hash', 'username', 'username_hash', 'password']

def load_key():
    with open(KEY_FILE, 'rb') as key_file:
        return key_file.read()

key = load_key()
cipher = Fernet(key)

def compute_hash(value):
    """Вычисление SHA-256 хэша строки."""
    return hashlib.sha256(value.encode()).hexdigest()

def get_users_df():
    """
    Загружает базу данных пользователей из CSV в DataFrame.
    """
    df = pd.read_csv(DATABASE_FILE)
    return df

def get_next_user_id(df):
    """
    Определяет следующий ID для нового пользователя.
    Если DataFrame пустой, возвращает базовый ID.
    """
    if not df.empty:
        return int(df['id'].max()) + 1
    else:
        return 270897  # ID первого пользователя

def register_user(email, username, password):
    """
    Регистрирует нового пользователя:
    - Проверяет корректность и уникальность email и username.
    - Шифрует email, username и password.
    - Добавляет новую запись в DataFrame и сохраняет в CSV.
    """
    # Вычисляем хэши для проверки уникальности
    email_hash = compute_hash(email)
    username_hash = compute_hash(username)
    
    # Загружаем DataFrame с данными пользователей
    df = get_users_df()

    # Проверка на существование email или username по их хэшам
    if email_hash in df['email_hash'].values:
        raise ValueError("Email already exists.")
    if username_hash in df['username_hash'].values:
        raise ValueError("Username already exists.")
    
    # Проверка обязательных полей и их формата
    errors = [
        ("Email, username, and password are required.", not all([email, username, password])),
        ("Password must be at least 8 characters long.", len(password) < 8),
        ("Username must be at least 3 characters long.", len(username) < 3),
        ("Invalid email format.", not match(r"[^@]+@[^@]+\.[^@]+", email)),
        ("Username can only contain letters, numbers, and underscores.", not match(r"^[a-zA-Z0-9_]+$", username)),
        ("Password can only contain letters, numbers, and underscores.", not match(r"^[a-zA-Z0-9_]+$", password)),
    ]
    for error_message, condition in errors:
        if condition:
            raise ValueError(error_message)

    # Определяем новый ID
    user_id = get_next_user_id(df)
    
    # Шифруем данные пользователя
    encrypted_email = cipher.encrypt(email.encode()).decode()
    encrypted_username = cipher.encrypt(username.encode()).decode()
    encrypted_password = cipher.encrypt(password.encode()).decode()

    # Формируем новую запись в виде словаря
    new_row = {
        'id': user_id,
        'email': encrypted_email,
        'email_hash': email_hash,
        'username': encrypted_username,
        'username_hash': username_hash,
        'password': encrypted_password
    }
    
    # Преобразуем словарь в DataFrame
    new_row_df = pd.DataFrame([new_row])

    # Объединяем существующий DataFrame с новым
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    # Сохраняем обновленный DataFrame в CSV
    df.to_csv(DATABASE_FILE, index=False)
    
    return True
