# !/usr/bin/env python2
# -*- coding: utf-8 -*-

import utils
import cv2
from functools import reduce

output_path = 'data/export'


def generate_set(size=200):
    """
    Generates a training set by loading all examples into memory, and resizing them
    :param size:
    :return:
    """

    global output_path

    label_paths = utils.get_file_paths('data/output/labels')
    example_paths = utils.get_file_paths('data/output/examples')

    output_path = "{0}/{1}x{1}/".format(output_path, size)

    # Make the path if it does not exist
    utils.make_path(output_path)

    names_list = []

    for i in range(len(label_paths)):
        names_list.append('{}'.format(i))

    img = cv2.imread(example_paths[20])

    mask_image(img, 230, 0.1)


def mask_image(image, size, layover=0.5):
    stride_size = (1-layover)*size
    sliding_space = image.shape[0] - size
    possible_factors = factors(sliding_space)
    stride_size = min(possible_factors, key=lambda factor_number: abs(factor_number-stride_size))

    iterations = int(sliding_space / stride_size)

    for i in range(iterations):
        y = i * stride_size
        for j in range(iterations):
            x = j * stride_size
            crop_img = image[y:y + size, x:x + size]
            cv2.imshow("cropped", crop_img)
            cv2.waitKey(0)


def factors(n):
    f = list(reduce(list.__add__, ([i, n//i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0)))
    return sorted(f)


if __name__ == "__main__":
    generate_set()
