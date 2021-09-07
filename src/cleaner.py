import os


if __name__ == '__main__':
    for folder, subfolder, filelist in os.walk('.\sliced'):
        if len(filelist) > 1:
            for file in filelist:
                if int(file[:3]) % 2 == 0:
                    os.remove(os.path.join(folder, file))
