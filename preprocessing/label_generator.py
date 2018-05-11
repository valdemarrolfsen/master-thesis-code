import threading
import uuid
from queue import Queue, Empty

from osgeo import gdal
import numpy as np
import os

from preprocessing import utils
from preprocessing.db import Db
import argparse
from shutil import copyfile


def setup():
    # Set ut the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', type=str, required=True, help='path to input file')
    ap.add_argument('-o', '--output', type=str, required=True, help='path for output file')
    ap.add_argument('-c', '--color', type=str, required=True, help='color value or color attribute in table')
    ap.add_argument('--prefix', type=str, default='', help='Prefix for files')
    ap.add_argument('--include-empty', type=bool, default=False, help='Include empty raster images')
    ap.add_argument('--binary', type=bool, default=False, help='Binary segmentation problem')
    ap.add_argument('--res', type=int, default=1000, help='Image resolution')
    ap.add_argument('--class-name', type=str, required=True,
                    help='The name of the class you are generating labels for')
    ap.add_argument(
        '-t',
        '--threads',
        type=int,
        default=8,
        help='the number of threads (defaults to 8)'
    )

    args = ap.parse_args()
    return args


def run(arguments):
    file_path = arguments['input']
    class_name = arguments['class_name']
    thread_count = arguments['threads']
    binary = arguments['binary']

    paths = ['train', 'test', 'val']
    sub_paths = ['examples', 'labels']
    for path in paths:
        for s in sub_paths:
            utils.make_path(os.path.join(arguments['output'], "{}/{}/{}/".format(path, s, arguments['class_name'])))

    tiff_files = utils.get_file_paths(file_path)
    total_files = len(tiff_files)
    print('Using class {}'.format(class_name))
    print('Found {} files'.format(total_files))
    if binary:
        print('Using binary')
    np.random.shuffle(tiff_files)
    q = Queue()
    for i, file in enumerate(tiff_files):
        q.put((file, i))

    print("Starting process with {} threads ".format(thread_count))

    for i in range(thread_count):
        # Create a new database connection for each thread.
        db = Db()
        db.connect()
        t = threading.Thread(target=work, args=(q, db, total_files, arguments))

        # Sticks the thread in a list so that it remains accessible
        t.daemon = True
        t.start()

    q.join()
    print("")


def is_raster_square(rast):
    # Use a virtual memory file, which is named like this
    i = uuid.uuid4()
    vsipath = '/vsimem/from_postgis' + str(i)
    gdal.FileFromMemBuffer(vsipath, bytes(rast))
    ds = gdal.Open(vsipath)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    # We accept the rasters that come out 1 px wrong
    min_thresh = rows - 1
    max_thresh = rows + 1
    if cols < min_thresh or cols > max_thresh:
        return False
    # Close and clean up virtual memory file
    ds = None
    gdal.Unlink(vsipath)
    return True


def is_raster_empty(rast):
    # Use a virtual memory file, which is named like this
    i = uuid.uuid4()
    vsipath = '/vsimem/from_postgis' + str(i)
    gdal.FileFromMemBuffer(vsipath, bytes(rast))
    ds = gdal.Open(vsipath)
    arr = np.array(ds.GetRasterBand(1).ReadAsArray())
    unique = np.unique(arr)

    if len(unique) < 2 and unique[0] == 0:
        return True

    # Close and clean up virtual memory file
    ds = None
    gdal.Unlink(vsipath)
    return False


def work(q, db, total_files, arguments):
    train_portion = 0.7
    val_portion = 0.2

    binary = arguments['binary']
    res = arguments['res']
    class_name = arguments['class_name']
    color = arguments['color']
    prefix = arguments['prefix']
    output_path = arguments['output']
    include_empty = arguments['include_empty']

    while not q.empty():
        try:
            file, i = q.get(True)
        except Empty:
            break

        utils.print_process(total_files - q.qsize(), total_files)
        min_x, min_y, max_x, max_y = utils.get_bounding_box_from_tiff(file)
        if min_x == -1:
            continue

        if binary:
            raster_records = db.get_binary_tiff(
                min_x,
                min_y,
                max_x,
                max_y,
                res,
                class_name,
                color
            )
        else:
            raster_records = db.get_tif_from_bbox(
                min_x,
                min_y,
                max_x,
                max_y,
                res,
                color
            )
        # Save the file to an unique id and add the correct file ending
        # are we adding to train, val or test?
        prog = (total_files - q.qsize()) / total_files

        filename = "{}{}-{}.{}".format(prefix, i, uuid.uuid4(), 'tif')

        if prog < train_portion:
            s = 'train'
        elif prog < train_portion + val_portion:
            s = 'val'
        else:
            s = 'test'

        examples_path = os.path.join(output_path, "{}/examples/{}/".format(s, class_name))
        labels_path = os.path.join(output_path, "{}/labels/{}/".format(s, class_name))

        path = os.path.join(labels_path, 'label_' + filename)

        if not raster_records:
            q.task_done()
            continue

        rast = raster_records[0][0]

        # Sometimes the raster is empty. We therefore have to save it as an empty raster
        if rast is None:
            q.task_done()
            continue

        if not is_raster_square(rast):
            q.task_done()
            continue

        if not include_empty and is_raster_empty(rast):
            q.task_done()
            continue

        copyfile(file, os.path.join(examples_path, filename))
        utils.save_file(rast, path)
        q.task_done()
    db.disconnect()
    q.task_done()


if __name__ == '__main__':
    argu = setup()
    run(argu)
