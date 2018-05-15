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
        for file in tqdm(files):
            im = os.path.join(path, file)
            img = cv2.imread(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ims.append(img / 255)
    print(np.mean(ims, axis=(0, 1, 2)))
    print(np.std(ims, axis=(0, 1, 2)))


if __name__ == '__main__':
    run()
