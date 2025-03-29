import csv
from cryptography.fernet import Fernet
from re import match

# Генерация или загрузка ключа шифрования
KEY_FILE = 'data/secret.key'

def load_key():
    with open(KEY_FILE, 'rb') as key_file:
        key = key_file.read()
    return key

key = load_key()
cipher = Fernet(key)

# Путь к базе данных
DATABASE_FILE = 'data/users.csv'

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
    # Проверки на валидность данных
    validations = [
        # Проверяем, что email, username и password не пустые
        (not email or not username or not password, "Email, username, and password are required."),
        # Проверяем на длину пароля и имени пользователя
        (len(password) < 8, "Password must be at least 8 characters long."),
        (len(username) < 3, "Username must be at least 3 characters long."),
        # Проверяем формат email, username и password
        (not match(r"[^@]+@[^@]+\.[^@]+", email), "Invalid email format."),
        (not match(r"^[a-zA-Z0-9_]+$", username), "Username can only contain letters, numbers, and underscores."),
        (not match(r"^[a-zA-Z0-9_]+$", password), "Password can only contain letters, numbers, and underscores."),
        # Проверяем, что email и username не существуют в базе данных
        (email in [row[1] for row in csv.reader(open(DATABASE_FILE))], "Email already exists."),
        (username in [row[2] for row in csv.reader(open(DATABASE_FILE))], "Username already exists.")
    ]

    # Проверяем каждую валидацию
    for condition, error_message in validations:
        if condition:
            raise ValueError(error_message)
    
    # Если все проверки пройдены, шифруем данные
    user_id = get_next_user_id()
    encrypted_email = cipher.encrypt(email.encode()).decode()
    encrypted_username = cipher.encrypt(username.encode()).decode()
    encrypted_password = cipher.encrypt(password.encode()).decode()

    with open(DATABASE_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, encrypted_email, encrypted_username, encrypted_password])

    return True