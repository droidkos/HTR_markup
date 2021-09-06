import os


for folder, subfolder, filelist in os.walk('.\sliced'):
    if len(filelist) > 1:
        for file in filelist:
            if int(file[:3]) % 2 == 0:
                # print('file to delete:', os.path.join(folder, file))
                os.remove(os.path.join(folder, file))
