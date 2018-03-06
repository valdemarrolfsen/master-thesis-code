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
            gdaltran_string = 'gdal_translate -of GTIFF -srcwin {} {} {} {} {} {}tiled_{}_{}.tif'.format(
                str(i), str(j), str(w), str(h), args.input_file, args.output_dir, str(i), str(j)
            )
            q.put(gdaltran_string)

    for i in range(args.threads):
        t = threading.Thread(target=work, args=(q, ))
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
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()

    if not args.input_file:
        print('Usage: tile.py --input_file filename --output_dir (optional)')
        sys.exit(0)
    run(args)
