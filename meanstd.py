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
        means.append(np.mean(ims, axis=(0, 1, 2)))

    print(np.mean(means, axis=0))
    print(np.std(ims, axis=(0, 1, 2)))


if __name__ == '__main__':
    run()
