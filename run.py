import threading
from queue import Queue, Empty

import os

import utils
from db import Db
import argparse
import shutil


file_path = None
output_path = None
thread_count = None
examples_path = None
labels_path = None
file_type = 'tif'


def setup():
    global file_path
    global output_path
    global thread_count
    global examples_path
    global labels_path

    # Set ut the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='path to input file')
    ap.add_argument('-o', '--output', required=True, help='path for output file')
    ap.add_argument(
        '-t',
        '--threads',
        required=True,
        type=int,
        default=10,
        help='the number of threads (defaults to 10)'
    )

    args = vars(ap.parse_args())

    # Get the file path
    file_path = args['input']
    output_path = args['output']
    thread_count = args['threads']

    labels_path = os.path.join(output_path, "labels/")
    examples_path = os.path.join(output_path, "examples/")

    utils.make_path(labels_path)
    utils.make_path(examples_path)


def run():
    global file_path
    global thread_count

    tiff_files = utils.get_file_paths(file_path)
    table_name = 'veg_flate'
    color_attribute = '255'
    total_files = len(tiff_files)
    print('Using table {}'.format(table_name))
    print('Found {} files'.format(total_files))

    q = Queue()
    for i, file in enumerate(tiff_files):
        q.put((file, i))

    print("Starting process with {} thread ".format(thread_count))

    for i in range(thread_count):
        # Create a new database connection for each thread.
        db = Db()
        db.connect()
        t = threading.Thread(target=work, args=(q, db, table_name, color_attribute))

        # Sticks the thread in a list so that it remains accessible
        t.daemon = True
        t.start()

    q.join()


def work(q, db, table_name, color_attribute):
    global file_type
    global examples_path
    global labels_path

    while not q.empty():
        try:
            file, i = q.get(False)
        except Empty:
            break

        min_x, min_y, max_x, max_y = utils.get_bounding_box_from_tiff(file)
        records = db.get_tif_from_bbox(
            min_x,
            min_y,
            max_x,
            max_y,
            table_name,
            color_attribute
        )

        # Save the file to an unique id and add the correct file ending
        filename = "{}.{}".format(i, file_type)

        path = os.path.join(labels_path, filename)
        width, height = utils.get_raster_size(file)

        if not records:
            utils.save_blank_raster(path, width, height)
            q.task_done()
            continue

        rast = records[0][0]

        # Sometimes the raster is empty. We therefore have to save it as an empty raster
        if rast is None:
            utils.save_blank_raster(path, width, height)
        else:
            utils.save_file(rast, path)

        # Save the original image to examples with the same name
        shutil.copy(file, os.path.join(examples_path, filename))

        q.task_done()


if __name__ == '__main__':
    setup()
    run()
