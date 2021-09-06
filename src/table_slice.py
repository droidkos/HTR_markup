import cv2
import numpy as np


def binarize(img):
    """
    Бинаризация изображения
    :param img: входное изображение (ndarray)
    :return: выходное изображение (ndarray)
    """
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin


def get_lines(orig_img, bin_img):
    """
    Получить вертикальные и горизонтальные границы ячеек
    :param  orig_img: исходное изображение (ndarray)
            bin_img: бинаризованное изображение (ndarray)
    :return:    img_vh: изображение из полученных границ на белом фоне (ndarray),
                bitnot: исходное изображение без полученных границ (ndarray)
    """
    # Размер ядра
    kernel_len = np.array(orig_img).shape[1] // 200
    # Определение вертикальных линий
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Определение горизонтальных линий
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # Ядро 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Получим новое изображение с вертикальными линиями
    ver_image = cv2.erode(bin_img, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(ver_image, ver_kernel, iterations=3)
    # Получим новое изображение с горизонтальными линиями
    hor_image = cv2.erode(bin_img, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(hor_image, hor_kernel, iterations=3)
    # Комбинируем вертикальные и горизонтальные линии в ноеое изображение,
    # присваивая им одинаковые веса
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    # Постобработка изображения
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(orig_img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    return img_vh, bitnot


def sort_contours(cnts, method="left-to-right"):
    """
    Функция для сортировки
    :param cnts: контуры для сортировки
    :param method: направление сортировки
    :return: отсортированные контуры
    """
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    else:
        reverse = False
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    cnts, bounding_boxes = zip(*sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts, bounding_boxes


def get_images(img_vh, bitnot, img_bin, w_min, h_min, h_max, debug=False):
    """
    Функция выполняет основную работу по выделению сегментов таблицы (ячеек) на изображении
    :param img_vh:  считанные границы таблицы (проще говоря, пустая таблица на белом фоне)
                    все изображения представляют собой двумерный массив numpy
    :param bitnot:  изображение с вырезанными границами таблицы
    :param img_bin: исходное изображение в черно-белом формате
    :param w_min:   не дает считать слишком маленькие контуры (побочка).
                    чем больше - тем более мелкие контуры допустимы. параметр для горизонтельных линий
    :param h_min:   то же, только для вертикальных линий
    :param h_max:   то же, только для слишком больших вертикальных линий
                    (дабы не считало за ячейку всю таблицу, к примеру)
    :param debug:   включает режим отладки - функция тогда
                    возвращает массив с границами для их визуальной оценки
    :return:        список сегментов изображения (ячейки таблицы)
    """
    # Определение и сортировка контуров
    bitnot = 255 - bitnot
    # img_vh = 255 - img_vh
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, bounding_boxes = sort_contours(contours, method="top-to-bottom")
    heights = [bounding_boxes[i][3] for i in range(len(bounding_boxes))]
    mean = np.mean(heights)

    image_h, image_w = img_bin.shape
    # Create list box to store all boxes in
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # if w > (image_w // w_min) and (image_h // h_min) < h < (image_h // h_max):
        if w > 100 and 500 > h > 50:
            image = cv2.rectangle(img_bin, (x, y), (x + w, y + h), (255, 0, 0), 20)
            box.append([x, y, w, h])

    # режим отладки возвращает массив с границами для их визуальной оценки
    if debug:
        try:
            return image
        except UnboundLocalError:
            print("Границ не найдено")

    # Создаем два списка для хранения строки и столбца, где расположены ячейки
    rows = []
    column = []
    j = 0
    # Сортировка bounding boxes
    for i in range(len(box)):
        if i == 0:
            column.append(box[i])
            previous = box[i]
        else:
            if box[i][1] <= previous[1] + mean / 2:
                column.append(box[i])
                previous = box[i]
                if i == len(box) - 1:
                    rows.append(column)
            else:
                rows.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    # Вычисление количества колонок
    countcol_max = 0
    for i in range(len(rows)):
        countcol = len(rows[i])
        if countcol > countcol_max:
            countcol_max = countcol

    arr = np.array(rows)
    try:
        arr = arr.transpose(1, 0, 2)
    except ValueError:
        print("Не удалось считать контуры. Вероятно, границы ячеек плохо пропечатаны")
    center = (arr[:, 0, 0] + arr[:, 0, 2]) / 2
    center.sort()

    finalboxes = []
    for i in range(len(rows)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(rows[i])):
            diff = abs(center - (rows[i][j][0] + rows[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(rows[i][j])
        finalboxes.append(lis)

    cropped_images = []
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            if len(finalboxes[i][j]) != 0:
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                 finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x + h, y:y + w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel, iterations=1)
                    erosion = cv2.erode(dilation, kernel, iterations=1)

                    cropped_images.append(255 - erosion)

    return cropped_images


def control(answers, cropped_images):
    """
    Проверка корректности формирования разметки
    :param answers: тексты (словосочетания)
    :param cropped_images: соответствующие им изображения (массивы numpy)
    :return:
    """
    assert 2 * len(answers) == len(cropped_images)
    min_dims = np.array([1e8, 1e8], dtype=int)
    for idx, arr in enumerate(cropped_images):
        if idx % 2 == 1:
            min_dims = np.minimum(min_dims, arr.shape)
    concat = []
    for idx, arr in enumerate(cropped_images):
        if idx % 2 == 1:
            concat.append(arr[:min_dims[0], :min_dims[1]])
    stack = np.vstack(concat)
    showme = cv2.resize(stack, (0, 0), fx=0.2, fy=0.2)
    did_write = cv2.imwrite('control.png', showme)
    assert did_write
    print("Контрольный файл записан")
    input("Проверь файл, нажми кнопку \n")
