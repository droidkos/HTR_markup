import os


def cleaner(folder):
    """
    :param folder: папка, где чистим
    Функция убирает изображения с печатным текстом из папки с нерезанными изображениями
    """
    for folder, subfolder, filelist in os.walk(folder):
        if len(filelist) > 1:
            for file in filelist:
                if int(file[:3]) % 2 == 0:
                    os.remove(os.path.join(folder, file))


if __name__ == '__main__':
    cleaner('.\sliced')
