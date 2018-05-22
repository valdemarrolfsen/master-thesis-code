import argparse
import os, sys
import threading

from osgeo import gdal
from queue import Queue, Empty


def run(args):
    dset = gdal.Open(args.input_file)
    width = dset.RasterXSize
    height = dset.RasterYSize
    print(width, 'x', height)
    tilesize = args.tile_size
    q = Queue()
    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            w = min(i + tilesize, width) - i
            h = min(j + tilesize, height) - j
            file_name = os.path.splitext(os.path.basename(args.input_file))[0]
            output_dir = os.path.join(args.output_dir, '{}_tiled_{}_{}.tif'.format(file_name, str(i), str(j)))
            gdaltran_string = 'gdal_translate -of GTIFF -srcwin {} {} {} {} {} {}'.format(
                str(i), str(j), str(w), str(h), args.input_file, output_dir
            )
            q.put(gdaltran_string)

    for i in range(args.threads):
        t = threading.Thread(target=work, args=(q,))
        t.daemon = True
        t.start()

    q.join()


def work(q):
    while not q.empty():
        try:
            gdaltrans_string = q.get(False)
        except Empty:
            break
        os.system(gdaltrans_string)
        q.task_done()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--tile-size', required=True, type=int, default=512)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()

    run(args)
