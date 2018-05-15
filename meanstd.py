import gc

import os
import cv2
import numpy as np
import random
from tqdm import tqdm

folders = ['bergen', 'bodo', 'oslo', 'stavanger', 'tromso', 'trondheim']
imdir = '/mnt/ekstern/rubval/{}/tiled-512x512/'


def run():
    means = []
    r_values = []
    g_values = []
    b_values = []

    files_per_iteration = 60
    files = []

    for folder in folders:
        path = imdir.format(folder)
        files_paths = os.listdir(path)
        absolute_paths = [os.path.join(path, file) for file in files_paths]
        files = files + absolute_paths

    print("Found {} files".format(len(files)))
    iterations = len(files)//files_per_iteration

    for i in tqdm(range(iterations)):
        ims = []
        current_files = files[i*files_per_iteration:(i+1)*files_per_iteration]

        for j, file in enumerate(current_files):
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255
            ims.append(img)
            gc.collect()

        ims = np.array(ims)
        r_values.append(ims[:][:][:][0].flatten())
        g_values.append(ims[:][:][:][1].flatten())
        b_values.append(ims[:][:][:][2].flatten())

        print(ims.shape)

        means.append(np.mean(ims, axis=(0, 1, 2)))

    print(np.mean(means, axis=0))
    print(np.std(r_values))
    print(np.std(g_values))
    print(np.std(b_values))


if __name__ == '__main__':
    run()
