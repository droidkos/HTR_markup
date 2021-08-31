import os
import numpy as np
import cv2

from src import markup


class DataProvider:

    def __init__(self, labels_file, image_folder, preprocess=False):
        self.labels = markup.read_source(labels_file)
        self.preprocess = preprocess
        image_list = os.listdir(image_folder)
        self.images = [os.path.join(image_folder, file) for idx, file in enumerate(image_list) if idx % 2 == 1]

    def __iter__(self):
        for lbl, img_file in zip(self.labels, self.images):
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            if self.preprocess:
                img = preprocess_image(img)
            yield lbl, img


def preprocess_image(img):
    # увеличит контраст
    pxmin = np.min(img)
    pxmax = np.max(img)
    imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

    # увеличить толщину линий
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.erode(imgContrast, kernel, iterations=1)
    return imgMorph


def createIAMCompatibleDataset(data_provider):
    """
    this function converts the passed dataset to an IAM compatible dataset
    """

    # подготовка файло и директорий
    f = open('../words.txt', 'w+')
    if not os.path.exists('../sub'):
        os.makedirs('../sub')
    if not os.path.exists('../sub/sub-sub'):
        os.makedirs('../sub/sub-sub')

    # конвертация данных в IAM-совместимый формат
    cnt = 0
    for label, image in data_provider:

        # write img
        cv2.imwrite('sub/sub-sub/sub-sub-%d.png' % cnt, image)

        # write filename, dummy-values and text
        line = 'sub-sub-%d' % cnt + ' X X X X X X X ' + label + '\n'
        f.write(line)

        cnt += 1


if __name__ == '__main__':
    dataProvider = DataProvider('../data/labels/orig_1.csv', 'sliced')
    createIAMCompatibleDataset(dataProvider)
