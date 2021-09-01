import os
import argparse

import cv2

import markup
import table_slice as ts

if __name__ == '__main__':
    # интерпретатор должен быть запущен в корневой папке проекта!

    parser = argparse.ArgumentParser()
    parser.add_argument("labels", choices=['blvrd', 'discipl', 'econ', 'journ', 'koms', 'mathstat'],
                        help='Название файла с текстом. Не указывать расширение!')
    parser.add_argument("images", help='Файл с отсканированным изображением')
    parser.add_argument("page", type=int, help='Номер страницы')
    parser.add_argument("gender", type=int, choices=[0,1], help="Пол: 1 - М, 0 - Ж")
    args = parser.parse_args()

    # аргументами должны быть только имена файлов, без путей
    # path дает кривые слеши, поэтому использую конкатенацию строк
    DST_IMG_FOLDER = './sliced/' + args.labels + '/' + args.images[:-4]  # убираем расширение файла
    # папка с готовыми (нарезанными) изображениями
    SRC_IMG_FOLDER = './data/images'  # папка с исходными (сканированными) изображениями
    SRC_LBL_FOLDER = './data/labels'  # папка с нарезанными текстами (csv)
    LABELS_FILE_PATH = SRC_LBL_FOLDER + '/' + args.labels + '.csv'  # путь к обрабатываемому файлу с ярлыками
    IMAGE_FILE_PATH = SRC_IMG_FOLDER + '/' + args.labels + '/' + args.images  # путь к фотке, которую нарезаем

    if not os.path.exists(DST_IMG_FOLDER):  # для каждой фотки - своя папка с нарезанными кусочками
        os.makedirs(DST_IMG_FOLDER)  # папка называется именем файла исходной фотки

    img = cv2.imread(IMAGE_FILE_PATH, 0)  # 0 для игнора цветовой палитры (читает ЧБ)
    if img is None:
        raise Exception("Не удалось прочитать исходный файл")  # т.к. opencv не выдает ошибок чтения

    # обработка изображений состоит из бинаризации
    # получения границ таблицы и непосредственно нарезки
    img_bin = ts.binarize(img)
    img_vh, bitnot = ts.get_lines(img, img_bin)
    cropped_images = ts.get_images(img_vh, bitnot, img_bin, w_min=10, h_min=25, h_max=5, debug=False)

    for num, cropped_img in enumerate(cropped_images):
        enhanced_img = markup.increase_contrast(cropped_img)
        filename = DST_IMG_FOLDER + '/' + str(num).zfill(3) + '.jpg'  # zfill делает названия 001, 002 и т.п.
        did_write = cv2.imwrite(filename, enhanced_img)
        # если не удалось записать файл, самостоятельно вызываем исключение
        if not did_write:
            raise Exception("Не удалось сохранить готовый файл")

    # получить из csv тексты с соответствующей страницы
    answers = markup.read_source(LABELS_FILE_PATH, args.page)

    # контроль
    ts.control(answers, cropped_images)

    # формирование и запись csv разметки (формат "текст" - "путь к картинке")
    markup.write_markup(LABELS_FILE_PATH, DST_IMG_FOLDER, args.labels, args.page, args.gender)
    print("Файл разметки сформирован")
