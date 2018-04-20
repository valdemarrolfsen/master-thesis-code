import argparse
import os
import threading
import cv2

import imgaug as ia
import imgaug as iaa
from queue import Queue, Empty

from keras_utils.prediction import geo_reference_raster, get_geo_frame
from preprocessing import utils

ia.seed(1)

processes = {
    'flipVertical': {'both': True, 'seq': iaa.Flipud(1)},

    'flipHorizontal': {'both': True, 'seq': iaa.Fliplr(1)},

    'blur': {'both': False, 'seq': iaa.GaussianBlur(sigma=(0.0, 3.0))},

    'sharpen': {'both': False, 'seq': iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))}
}

folders = ['test', 'train', 'val']


def run(args):
    input_folder = args.input_folder
    output_path = args.output_path
    class_name = args.class_name

    for key in processes.keys():
        for folder in folders:
            examples_path = os.path.join(input_folder, folder, 'examples', class_name)
            labels_path = os.path.join(input_folder, folder, 'labels', class_name)
            examples_output_path = os.path.join(output_path, folder, 'examples', class_name)
            labels_output_path = os.path.join(output_path, folder, 'labels', class_name)

            examples = utils.get_file_paths(examples_path)
            labels = utils.get_file_paths(labels_path)

            total_files = len(examples)

            q = Queue()
            for i, file in enumerate(examples):
                q.put((file, labels[i], i))

            for i in range(args.thread_count):
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

        example_aug_save_path = os.path.join(examples_output_path, '{}-{}'.format(key, example_path))
        label_aug_save_path = os.path.join(labels_output_path, '{}-{}'.format(key, label_path))

        cv2.imwrite(example_aug_save_path, example_aug)
        cv2.imwrite(label_aug_save_path, label_aug)

        try:
            # Get coordinates for corresponding image
            ulx, scalex, skewx, uly, skewy, scaley = get_geo_frame(example)

            # Geo reference newly created raster
            geo_reference_raster(
                example_aug_save_path,
                [ulx, scalex, skewx, uly, skewy, scaley]
            )

            geo_reference_raster(
                label_aug_save_path,
                [ulx, scalex, skewx, uly, skewy, scaley]
            )

        except ValueError as e:
            print("Was not able to reference image at path: {}".format(example_aug_save_path))

        q.task_done()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-path', type=str, required=True, help='Path to file folder')
    ap.add_argument('--output-path', type=str, required=True, help='Path to output folder')
    args = ap.parse_args()

    run(args)
