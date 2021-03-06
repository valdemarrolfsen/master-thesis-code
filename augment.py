import argparse
import os
import threading
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

from queue import Queue, Empty

from preprocessing import utils

ia.seed(1)

processes = {
    'flipVertical': {'both': True, 'seq': iaa.Flipud(1)},

    'flipHorizontal': {'both': True, 'seq': iaa.Fliplr(1)},

    'blur': {'both': False, 'seq': iaa.GaussianBlur(sigma=(0.0, 3.0))},

    'sharpen': {'both': False, 'seq': iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))}
}

folders = ['train']


def run(arguments):
    input_folder = arguments.input_path
    output_path = arguments.output_path
    class_name = arguments.class_name

    for key in processes:

        print('Starting augmentation: {}'.format(key))

        for folder in folders:

            print('Starting augmentation {} for folder: {}'.format(key, folder))

            examples_path = os.path.join(input_folder, folder, 'examples', class_name)
            labels_path = os.path.join(input_folder, folder, 'labels', class_name)
            examples_output_path = os.path.join(output_path, folder, 'examples', class_name)
            labels_output_path = os.path.join(output_path, folder, 'labels', class_name)

            if not os.path.exists(examples_output_path):
                os.makedirs(examples_output_path)

            if not os.path.exists(labels_output_path):
                os.makedirs(labels_output_path)

            print('Using path: {}'.format(examples_path))

            examples = utils.get_file_paths(examples_path)
            labels = utils.get_file_paths(labels_path)

            total_files = len(examples)

            print('Found {} files'.format(total_files))

            q = Queue()
            for i, file in enumerate(examples):
                q.put((file, labels[i], i))

            print("Starting process with {} threads ".format(arguments.thread_count))

            for i in range(arguments.thread_count):
                t = threading.Thread(target=work, args=(q, examples_output_path, labels_output_path, key, total_files))

                # Sticks the thread in a list so that it remains accessible
                t.daemon = True
                t.start()

            q.join()


def work(q, examples_output_path, labels_output_path, key, total_files=0):
    while not q.empty():
        try:
            example_path, label_path, i = q.get(False)
        except Empty:
            break

        current_augmentation = processes[key]
        seq = current_augmentation['seq']
        utils.print_process(total_files - q.qsize(), total_files)

        example = cv2.imread(example_path)
        label = cv2.imread(label_path)

        example_aug = seq.augment_image(example)
        label_aug = label

        if current_augmentation['both']:
            label_aug = seq.augment_image(label)

        example_aug_save_path = os.path.join(examples_output_path, '{}-{}'.format(key, example_path.split('/')[-1]))
        label_aug_save_path = os.path.join(labels_output_path, '{}-{}'.format(key, label_path.split('/')[-1]))

        cv2.imwrite(example_aug_save_path, example_aug)
        cv2.imwrite(label_aug_save_path, label_aug)

        q.task_done()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-path', type=str, required=True, help='Path to file folder')
    ap.add_argument('--output-path', type=str, required=True, help='Path to output folder')
    ap.add_argument('--class-name', type=str, required=True, help='Class name')
    ap.add_argument('--thread-count', type=int, default=8, help='Number of worker threads to spawn')
    arguments = ap.parse_args()

    run(arguments)
