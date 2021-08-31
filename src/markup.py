import re
import csv
import os
import cv2
import numpy as np


def write_source(input_path, output_path, line_cnt, words_per_line=3, replace=False):
    """
    Формирует csv файл из простого текстового файла (файл-источник)
    Данный файл необходим для формирования labels разметки
    :param input_path: путь к txt файлу источника
    :param output_path: путь к csv файлу приемника
    :param line_cnt: ограничение на число выходных строк
    :param words_per_line: количество слов в строке
    :param replace: заменять ли файл-приемник, если он существует
    :return: None
    """
    if replace and os.path.isfile(output_path):
        os.remove(output_path)
    elif os.path.isfile(output_path):
        raise Exception("Алярм! Выходной файл уже существует!")

    with open(input_path, encoding='utf-8') as file:
        text = file.read().lower()
        text = text.replace('ниу', ' ')
        text = text.replace('вшэ', ' ')
        text = re.sub(r'[^а-я ]+', ' ', text)

    words = text.split()
    words_total = len(words) - 3
    pairs = []

    while line_cnt > 0:
        rand = np.random.randint(0, words_total)
        if len(words[rand]) < 3 and len(words[rand + 1]) < 3:
            continue
        elif len(words[rand + 1]) < 3 and len(words[rand + 2]) < 3:
            continue
        elif len(words[rand]) < 3 and len(words[rand + 2]) < 3:
            continue

        pairs.append(' '.join(words[rand: rand + words_per_line]))
        line_cnt -= 1

    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        for pair in pairs:
            writer.writerow([pair])


def read_source(path, page):
    """
    Вспомогательная функция для чтения сформированного csv-файла (labels будущей разметки)
    :param nrows: сколько строк считывать
    :param offset: смещение (пропуск N строк)
    :param path: путь к файлу
    :return: список из словосочетаний
    """
    nrows = 21
    offset = (page - 1) * nrows  # на каждой странице у нас по 20 словосочетаний
    n = 0
    with open(path, newline='') as file:
        answers = []
        for _ in range(offset):  # skip first 10 rows
            next(file)
        for row in csv.reader(file):
            answers.append(row[0])  # у нашего файла 1 столбец
            n += 1
            if n >= nrows:
                break
    return answers


def write_markup(labels_path, image_folder, labels_filename, page):
    """
    Принимает csv файл с labels и папку
    Формирование файла разметки для обучения модели
    :param labels_path: путь к файлу с текстами строк
    :param page: номер страницы (определяет смещение в csv-файле)
    :param labels_filename: называние файла с текстами строк
    :param image_folder: путь к папке с нарезанными изображениями
    :return:
    """
    markup_file_path = './markup/markup_' + labels_filename + '.csv'

    labels = read_source(labels_path, page)
    image_list = os.listdir(image_folder)
    # половина ячеек таблицы - печатный текст (исходник)
    # такие ячейки пропускаем, берем только нечетные
    images = [image_folder + '/' + file for idx, file in enumerate(image_list) if idx % 2 == 1]

    assert labels, "Список текстов пустой"
    assert images, "Директория с нарезанными картинками пуста"
    assert len(labels) == len(images), "Несовпадение длин массивов картинок и текстов"

    # если файла не существует (пишем в него первый раз) - добавляем заголовок
    if not os.path.exists(markup_file_path):
        with open(markup_file_path, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(('label', 'image'))

    # после этого добавляем в файл обработанные строки
    with open(markup_file_path, "a", newline='') as file:
        writer = csv.writer(file)
        for pair in zip(labels, images):
            writer.writerow(pair)


def increase_contrast(img):
    """
    Увеличивает контрастность и толщину линий для улучшения качества распознавания
    :param img: исходное изображение
    :return: обработанное изображение
    """
    pxmin = np.min(img)
    pxmax = np.max(img)
    imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.erode(imgContrast, kernel, iterations=1)
    return imgMorph
