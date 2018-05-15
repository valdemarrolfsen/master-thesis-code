import gc

import os
import cv2
import numpy as np
from tqdm import tqdm

folders = ['train', 'val', 'test']
imdir = '/mnt/ekstern/rubval/buildings/{}/examples/buildings/'


def run():
    ims = []
    for folder in folders:
        path = imdir.format(folder)
        files = os.listdir(path)
        for i, file in enumerate(tqdm(files)):
            if i > 1000:
                break
            im = os.path.join(path, file)
            img = cv2.imread(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255
            ims.append(img)
            gc.collect()
    print(np.mean(ims, axis=(0, 1, 2)))
    print(np.std(ims, axis=(0, 1, 2)))


if __name__ == '__main__':
    run()
