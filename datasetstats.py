import argparse

from preprocessing import utils
from preprocessing.db import Db

global COUNT


def parse_args():
    # Set ut the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, required=True, help='path to input files')
    ap.add_argument('--class-name', type=str, required=True, help='class name')

    args = ap.parse_args()
    return args


def run(arguments):
    global COUNT
    COUNT = 0
    file_path = arguments.input
    class_name = arguments.class_name
    tiff_files = utils.get_file_paths(file_path)

    db = Db()
    db.connect()
    for file in tiff_files:
        min_x, min_y, max_x, max_y = utils.get_bounding_box_from_tiff(file)
        if min_x == -1:
            continue
        count = db.count_features(
            min_x,
            min_y,
            max_x,
            max_y,
            class_name
        )
        COUNT += count

    print('Count: {}'.format(COUNT))


if __name__ == '__main__':
    arguments = parse_args()
    run(arguments)
