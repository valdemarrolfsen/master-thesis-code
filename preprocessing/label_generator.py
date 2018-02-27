import threading
from queue import Queue, Empty

import os

import utils
from db import Db
import argparse
from shutil import copyfile

file_path = None
output_path = None
thread_count = None
examples_path = None
labels_path = None
color_attribute = None
table_name = None
file_type = 'tif'


def setup():
    global file_path
    global output_path
    global thread_count
    global examples_path
    global labels_path
    global color_attribute
    global table_name

    # Set ut the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='path to input file')
    ap.add_argument('-o', '--output', required=True, help='path for output file')
    ap.add_argument('-c', '--color', required=True, help='color value or color attribute in table')
    ap.add_argument('-n', '--table', required=True, help='table name')
    ap.add_argument(
        '-t',
        '--threads',
        type=int,
        default=10,
        help='the number of threads (defaults to 10)'
    )

    args = vars(ap.parse_args())

    # Get the file path
    file_path = args['input']
    output_path = args['output']
    thread_count = args['threads']
    color_attribute = args['color']
    table_name = args['table']
    labels_path = os.path.join(output_path, "labels/")
    examples_path = os.path.join(output_path, "examples/")

    utils.make_path(labels_path)
    utils.make_path(examples_path)


def run():
    global file_path
    global thread_count
    global color_attribute
    global table_name
    tiff_files = utils.get_file_paths(file_path)
    total_files = len(tiff_files)
    print('Using table {}'.format(table_name))
    print('Found {} files'.format(total_files))

    q = Queue()
    for i, file in enumerate(tiff_files):
        q.put((file, i))

    print("Starting process with {} threads ".format(thread_count))

    for i in range(thread_count):
        # Create a new database connection for each thread.
        db = Db()
        db.connect()
        t = threading.Thread(target=work, args=(q, db, table_name, color_attribute, total_files))

        # Sticks the thread in a list so that it remains accessible
        t.daemon = True
        t.start()

    q.join()
    print("")


def work(q, db, table_name, color_attribute, total_files=0):
    global file_type
    global examples_path
    global labels_path

    while not q.empty():
        try:
            file, i = q.get(False)
        except Empty:
            break

        utils.print_process(total_files - q.qsize(), total_files)
        min_x, min_y, max_x, max_y = utils.get_bounding_box_from_tiff(file)
        if min_x == -1:
            continue
        raster_records = db.get_tif_from_bbox(
            min_x,
            min_y,
            max_x,
            max_y,
            table_name,
            color_attribute
        )
        # Save the file to an unique id and add the correct file ending
        filename = "{}.{}".format(i, file_type)

        copyfile(file, os.path.join(examples_path, filename))

        path = os.path.join(labels_path, filename)
        width, height = utils.get_raster_size(file)

        if not raster_records:
            utils.save_blank_raster(path, width, height)
            q.task_done()
            continue

        rast = raster_records[0][0]

        # Sometimes the raster is empty. We therefore have to save it as an empty raster
        if rast is None:
            utils.save_blank_raster(path, width, height)
        else:
            utils.save_file(rast, path)

        # Get the geojson for the geometries
        geojson_record = db.get_geojson_from_bbox(
            min_x,
            min_y,
            max_x,
            max_y,
            table_name,
        )
        geojson = geojson_record[0][0]
        filename = "{}.{}".format(i, 'geojson')
        path = os.path.join(labels_path, filename)
        if not geojson:
            # Save empty json file
            utils.save_json('{}', path)
            continue
        utils.save_json(geojson, path)
        q.task_done()


if __name__ == '__main__':
    setup()
    run()