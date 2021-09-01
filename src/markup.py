import re
import csv
import os
import argparse
import cv2
import numpy as np


def write_source(input_path, line_cnt, words_per_line=3):
    """
    Формирует csv файл из простого текстового файла (файл-источник)
    Данный файл необходим для формирования labels разметки
    :param input_path: путь к txt файлу источника
    :param line_cnt: ограничение на число выходных строк
    :param words_per_line: количество слов в строке
    :return: None
    """
    with open(input_path, encoding='utf-8') as file:
        text = file.read().lower()
        text = text.replace('ниу', ' ')
        text = text.replace('вшэ', ' ')
        text = re.sub(r'[^а-я ]+', ' ', text)

    words = text.split()
    words_total = len(words) - 3

    while line_cnt > 0:
        rand = np.random.randint(0, words_total)
        if len(words[rand]) < 3 and len(words[rand + 1]) < 3:
            continue
        elif len(words[rand + 1]) < 3 and len(words[rand + 2]) < 3:
            continue
        elif len(words[rand]) < 3 and len(words[rand + 2]) < 3:
            continue

        concat = ' '.join(words[rand: rand + words_per_line])
        if len(concat) > 20:
            continue

        line_cnt -= 1
        yield concat


def read_source(path, page):
    """
    Вспомогательная функция для чтения сформированного csv-файла (labels будущей разметки)
    :param nrows: сколько строк считывать
    :param offset: смещение (пропуск N строк)
    :param path: путь к файлу
    :return: список из словосочетаний
    """
    nrows = 20
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


if __name__ == '__main__':
    # интерпретатор должен быть запущен в корневой папке проекта!
    # формирование исходного набора строк для разметки
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=['blvrd', 'discipl', 'econ', 'journ', 'koms', 'mathstat'],
                        help='Название файла с текстом. Не указывать расширение!')
    parser.add_argument("-p", "--pages", default=200, type=int, help="Кол-во страниц")
    parser.add_argument("--rewrite", action="store_true", help="Флаг перезаписи файлов")
    args = parser.parse_args()

    line_cnt = args.pages * 20  # 20 строк на странице
    SRC_TXT_FOLDER = './data/txt'
    TXT_FILE_PATH = SRC_TXT_FOLDER + '/' + args.source + '.txt'
    SRC_LBL_FOLDER = './data/labels'  # папка с нарезанными текстами (csv)
    LABELS_FILE_PATH = SRC_LBL_FOLDER + '/' + args.source + '.csv'  # путь к обрабатываемому файлу с ярлыками

    if os.path.isfile(LABELS_FILE_PATH) and not args.rewrite:
        raise Exception("Осторожно, файл уже существует. Для перезаписи укажи флаг --rewrite")

    with open(LABELS_FILE_PATH, "w", newline='') as f:
        writer = csv.writer(f)
        for pair in write_source(TXT_FILE_PATH, line_cnt=line_cnt):
            writer.writerow([pair])
