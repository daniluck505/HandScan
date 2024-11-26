# HandScan

Этот проект представляет собой решение задачи распознавания рукописного текста. На данный момент реализовано распознавание отдельных рукописных слов, а в будущем проект будет дополнен функцией детекции слов на изображении для создания Telegram бота, переводящего фотографии рукописного текста в обычный текст.

## Функционал
### Реализовано:
- **Распознавание отдельных рукописных слов:** модель принимает изображение одного слова и возвращает его текстовую интерпретацию.
- **Модульная структура:** код разделён на логически независимые модули для удобства разработки и тестирования.

### В планах:
- **Детекция слов на изображении:** распознавание всех слов на фотографии с их координатами.
- **Telegram-бот:** удобный пользовательский интерфейс для загрузки изображений и получения распознанного текста.

---

## Структура проекта
### Основные файлы:
```
OCR-Handwriting-Recognition
├── config.yml               # Конфигурация для обучения и предсказаний модели 
├── train.py                 # Скрипт для обучения модели 
├── work.ipynb               # Ноутбук для оценки модели и выполнения предсказаний 
├── model.py                 # Архитектуры модели для распознавания текста 
├── trainer.py               # Класс для обучения, валидации и логирования 
├── dataset.py               # Классы для работы с датасетами (предобработка, аугментации) 
├── utils.py                 # Вспомогательные функции для обработки данных, метрик и логирования 
├── requirements.txt         # Список зависимостей для проекта 
├── README.md                # Описание проекта и инструкция по использованию 
└── data_1 
    ├── train                # Папка с тренировочными данными (изображения в формате PNG) 
    └── test                 # Папка с тестовыми данными (изображения в формате PNG) 
```
---

## Технологии
- **PyTorch:** основная библиотека для разработки и обучения моделей машинного обучения.
- **Python:** язык программирования, используемый для реализации всего функционала.
- **yaml:** для хранения параметров конфигурации.
- **Telegram Bot API:** планируется для создания бота.

---

## Использование
### 1. Подготовка окружения:
- Установите зависимости из файла `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Настройка конфигурации:
- Измените файл `config.yml` в соответствии с вашими данными и параметрами.

### 3. Обучение модели:
- Запустите обучение с помощью:
  ```bash
  python train.py
  ```

### 4. Тестирование модели:
- Используйте ноутбук `work.ipynb` для оценки качества модели и выполнения предсказаний.

### Ссылки:
- Dataset https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset

---

## Пример использования
После завершения обучения модели можно загрузить изображение рукописного слова и получить его текстовую интерпретацию. 

Пример кода предсказания будет добавлен после реализации API модели.

---

## Планируемые обновления
- Разработка модуля детекции слов на изображении.
- Интеграция с Telegram для упрощения работы пользователей.
