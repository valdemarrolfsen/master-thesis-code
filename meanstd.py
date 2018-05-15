import os
import cv2
import random
import numpy as np
from tqdm import tqdm

folders = ['train', 'val', 'test']
imdir = '/mnt/ekstern/rubval/buildings/{}/examples/buildings/'


def run():
    ims = []
    files = []
    for folder in folders:
        path = imdir.format(folder)
        files += os.listdir(path)
    files = random.sample(files, 5000)
    for file in tqdm(files):
        path = imdir.format(folder)
        im = os.path.join(path, file)
        img = cv2.imread(im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ims.append(img / 255)
    print(np.mean(ims, axis=(0, 1, 2)))
    print(np.std(ims, axis=(0, 1, 2)))


if __name__ == '__main__':
    run()
