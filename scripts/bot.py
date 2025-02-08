
from fastbook import load_learner  

from aiogram import Bot
from aiogram import Dispatcher
from aiogram import types
from aiogram.filters import Command
from aiogram import F
import os
import aiosqlite
import logging
from datetime import datetime
import torch
from torchvision import models, transforms
from PIL import Image
import io



# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Необходимо получить токен у бота (@BotFather)
API_TOKEN = "______"

# Создаём объекты бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Путь к базе данных SQLite
DB_NAME = "habitat_monitoring.db"

# Создаём папку для сохранения фото
os.makedirs("photos", exist_ok=True)


# Загрузка добученной на fastai модели ResNet50 
model_path = 'russian_birds.pkl' # Путь к сохраненной модели
model = load_learner(model_path) # Загружаем модель


# Загрузка списка классов из файла с указанием кодировки UTF-8
classes={}
classes_dict_path='russian_birds.txt'
with open(classes_dict_path, encoding='utf-8') as f:
    classes_lines = [line.strip() for line in f.readlines()]
for cls in classes_lines:
    classes[cls.split(':')[0].strip()]=cls.split(':')[1].strip()    

# Функция для классификации изображений
def classify_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        pred_class, pred_idx, probs = model.predict(img)
        class_name = classes[pred_class]
        probability = float(probs[pred_idx])
        return pred_class, class_name, probability
    except Exception as e:
        logger.error(f"Ошибка при классификации изображения: {e}")
        return None, None,None




# Функция для создания таблицы
async def create_table():
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL DEFAULT 0,
                photo_id TEXT NOT NULL,
                photo_path TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                file_exists INTEGER DEFAULT 1,
                classified INTEGER DEFAULT 0,
                classification_success INTEGER DEFAULT 0,
                classification_1_model TEXT,
                classification_1_class TEXT,
                classification_1_probability REAL,
                classification_2_model TEXT,
                classification_2_class TEXT,
                classification_2_probability REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        await db.commit()

# Функция для миграции базы данных
async def migrate_db():
    async with aiosqlite.connect(DB_NAME) as db:
        # Проверяем, существует ли таблица
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='photos'")
        table_exists = await cursor.fetchone()
        if table_exists:
            # Создаём временную таблицу с новой структурой
            await db.execute('''
                CREATE TABLE IF NOT EXISTS new_photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL DEFAULT 0,
                    photo_id TEXT NOT NULL,
                    photo_path TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    file_exists INTEGER DEFAULT 1,
                    classified INTEGER DEFAULT 0,
                    classification_success INTEGER DEFAULT 0,
                    classification_1_model TEXT,
                    classification_1_class TEXT,
                    classification_1_probability REAL,
                    classification_2_model TEXT,
                    classification_2_class TEXT,
                    classification_2_probability REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            await db.commit()
            # Переносим данные из старой таблицы в новую
            await db.execute('''
                INSERT INTO new_photos (
                    id, user_id, photo_id, photo_path, latitude, longitude
                )
                SELECT 
                    id, 0, photo_id, photo_path, latitude, longitude 
                FROM photos
            ''')
            await db.commit()
            # Удаляем старую таблицу
            await db.execute('DROP TABLE photos')
            await db.commit()
            # Переименовываем новую таблицу в старую
            await db.execute('ALTER TABLE new_photos RENAME TO photos')
            await db.commit()
        else:
            # Если таблицы нет, создаём её с новой структурой
            await create_table()

# Обработчик команды /start
@dp.message(Command("start"))
async def start_command(message: types.Message):
    await message.answer("Привет! Я бот для мониторинга ареалов обитания. Отправь мне фотографию и геометку!")

# Обработчик для фотографий
@dp.message(F.content_type == "photo")
async def handle_photo(message: types.Message):
    try:
        user_id = message.from_user.id  # ID пользователя
        photo_id = message.photo[-1].file_id  # ID фотографии
        # Проверяем, существует ли уже такая фотография в базе
        async with aiosqlite.connect(DB_NAME) as db:
            cursor = await db.execute('''
                SELECT photo_id FROM photos WHERE photo_id = ? AND user_id = ?
            ''', (photo_id, user_id))
            existing_photo = await cursor.fetchone()
            if existing_photo:
                await message.answer("Эта фотография уже была загружена. Пожалуйста, отправьте другую.")
                return  # Прерываем выполнение функции
        # Если фотография новая, сохраняем её
        photo_file = await bot.get_file(photo_id)
        photo_extension = photo_file.file_path.split('.')[-1]
        photo_path = f"photos/{photo_id}.{photo_extension}"
        await bot.download_file(photo_file.file_path, photo_path)
        # Добавляем запись в базу данных
        async with aiosqlite.connect(DB_NAME) as db:
            await db.execute('''
                INSERT INTO photos (user_id, photo_id, photo_path)
                VALUES (?, ?, ?)
            ''', (user_id, photo_id, photo_path))
            await db.commit()
        await message.answer("Фото сохранено! Теперь отправь геометку.")
        
        # Классификация изображения
        class_index, class_name, probability = classify_image(photo_path)
        if class_index is not None:
            if probability>0.8:
                cls_success=1
            else:
                cls_success=0
            async with aiosqlite.connect(DB_NAME) as db:
                await db.execute('''
                    UPDATE photos 
                    SET classified = 1, 
                        classification_success = ?, 
                        classification_1_model = 'ResNet50', 
                        classification_1_class = ?, 
                        classification_1_probability = ?
                    WHERE photo_id = ? AND user_id = ?
                ''', (cls_success, class_name, probability ,photo_id, user_id))
                await db.commit()
            if cls_success==1:    
                await message.answer(f"Это: {class_name} с вероятностью {round(probability*100,2)}%")
            else:
                await message.answer(f"Я не смог классифицировать эту птицу.")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await message.answer("Ошибка при сохранении фото")

# Обработчик для геолокаций
@dp.message(F.content_type == "location")
async def handle_location(message: types.Message):
    try:
        user_id = message.from_user.id  # ID пользователя
        latitude = message.location.latitude
        longitude = message.location.longitude
        # Получаем последнее сохраненное фото пользователя без геометки
        async with aiosqlite.connect(DB_NAME) as db:
            cursor = await db.execute('''
                SELECT id FROM photos 
                WHERE user_id = ? AND latitude IS NULL 
                ORDER BY id DESC 
                LIMIT 1
            ''', (user_id,))
            last_photo = await cursor.fetchone()
            if last_photo:
                await db.execute('''
                    UPDATE photos 
                    SET latitude = ?, longitude = ? 
                    WHERE id = ?
                ''', (latitude, longitude, last_photo[0]))
                await db.commit()
                await message.answer(f"Геометка сохранена: {latitude}, {longitude}")
            else:
                await message.answer("Сначала отправьте фото")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await message.answer("Ошибка при сохранении геометки")

# Запуск бота
async def main():
    await migrate_db()  # Выполняем миграцию базы данных
    await create_table()  # Создаем таблицу, если её нет
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
