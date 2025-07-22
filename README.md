# Практическая работа в "Открытая мобильная работа"

## Задание

Нужно найти и реализовать легковесный вариант распознавания лица на C++.
Можно поискать готовые библиотеки, но желательно иметь минимум зависимостей.
Должно быть 2 режима:
- инициализация лица и сохранение в некую БД (файл)
- распознавание лица и поиск в базе сохраненных -> Результат true/false.
  В обоих режимах входные данные - видеопоток.

## Функционал проекта

Проект представляет собой легковесный вариант распознавания лиц в видеопотоке на С++.

В данном проекте реализовано два режима работы, определение режима работы происходит через аргументы командной строки:

### 1. Инициализация лица
    
Для режима регистрации лица реализован ниже представленный функцинонал: 

- Распознавание лица в видеопотоке
- Извлечение вектора признаков (embedding) с помощью MobileFaceNet
- Сохранение полученного вектора и данные пользователя в базу данных, на данном этапе JSON-файл

### 2. Распознавание лица

Для режима распознавания лица реализован ниже представленный функцинонал:

- Распознавание лица в видеопотоке
- Сравнивает полученный вектор признаков с базой данных
- В качестве результата выводит true/false ("Access denied" / "Access allowed").

## Назначение проекта

**Цель проекта** — реализация простой и производительной системы распознавания лиц на C++ с минимальными зависимостями.

- Внедрение в ОС АВРОРА
- Расширения для дальнейших проектов

## Использованные зависимости

- **OpenCV** - обработка видеопотока и запуск ONNX-моделей
- **nlohmann/json** - работа с JSON-базой данных
- **ONNX** - формат нейросетевых моделей:
  1. YuNet - модель детекции лиц
  2. MobileFaceNet - модель извлечения признаков

## Блок-схема алгоритма

![diagram.png](materials/diagram.png)


## Структура проекта

![structure.png](materials/structure.png)

# Сборка

### Зависимости

- CMake >= 3.10
- OpenCV >= 4.5.4 (используется FaceDetectorYN)
- Компилятор C++ с поддержкой C++17

### Клонирование

```bash
git clone --recurse-submodules https://github.com/lenchik-en/facedetection
cd facedetection
```

### Сборка проекта

```bash
mkdir -p build
cd build
cmake ..
make -j8
```

### Запуск

```bash
./main \
  --model=models/face_detection_yunet_2023mar.onnx \
  --vm=models/MobileFaceNet.onnx \
  --database=data/face_db.json \
  --mode=identify \
```

### Возможности

- `--mode=register`: сохранить новое лицо в базу (ввод имени вручную)
- `--mode=identify`: распознать лицо на кадре
- `--h=help`: Вывод сообщения

## OpenCV
Licensed under Apache License 2.0
https://github.com/opencv/opencv/blob/4.x/LICENSE

## nlohmann/json
Licensed under MIT
https://github.com/nlohmann/json/blob/develop/LICENSE.MIT

## MobileFaceNet
Licensed under MIT 
https://github.com/foamliu/MobileFaceNet/blob/master/LICENSE

---
MIT License

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.