import gc

import os
import cv2
import numpy as np
import random
from tqdm import tqdm

folders = ['train', 'val', 'test']
imdir = '/mnt/ekstern/rubval/buildings/{}/examples/buildings/'


def run():
    means = []
    r_values = []
    g_values = []
    b_values = []
    for i in tqdm(range(100)):
        ims = []
        for folder in folders:
            path = imdir.format(folder)
            files = os.listdir(path)
            files = random.sample(files, 20)
            for i, file in enumerate(files):
                im = os.path.join(path, file)
                img = cv2.imread(im)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255
                ims.append(img)
                gc.collect()

        ims = np.array(ims)
        r_values.append(ims[:][:][:][0].flatten())
        g_values.append(ims[:][:][:][1].flatten())
        b_values.append(ims[:][:][:][2].flatten())

        means.append(np.mean(ims, axis=(0, 1, 2)))

    print(np.mean(means, axis=0))
    print(np.std(r_values))
    print(np.std(g_values))
    print(np.std(b_values))


if __name__ == '__main__':
    run()
