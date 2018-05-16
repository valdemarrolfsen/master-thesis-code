import gc
from queue import Queue

import os
from PIL import Image
import numpy as np
import random
import threading
from tqdm import tqdm

from preprocessing import utils
from preprocessing.utils import contains_zero_value

folders = ['bergen', 'bodo', 'oslo', 'stavanger', 'tromso', 'trondheim']
imdir = '/mnt/ekstern/rubval/{}/tiled-512x512/'


def delete():
    files = []
    for folder in folders:
        path = imdir.format(folder)
        files_paths = os.listdir(path)
        absolute_paths = [os.path.join(path, file) for file in files_paths]
        files = files + absolute_paths
    print("Found {} files".format(len(files)))

    q = Queue()
    for i, file in enumerate(files):
        q.put((file, i))

    for i in range(8):
        t = threading.Thread(target=work, args=(q, len(files)))
        # Sticks the thread in a list so that it remains accessible
        t.daemon = True
        t.start()

    q.join()


def work(q, nb_files):
    while not q.empty():
        utils.print_process(nb_files - q.qsize(), nb_files)
        file, i = q.get()
        if contains_zero_value(file):
            os.remove(file)

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
            img = np.array(Image.open(file))
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

