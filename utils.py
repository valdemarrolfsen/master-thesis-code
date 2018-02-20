import json
import os
import numpy as np
import errno
import random
from collections import defaultdict
from osgeo import gdal
from scipy.misc import imresize, imread
from scipy.ndimage import zoom


def data_generator_s31(datadir='', nb_classes=None, batch_size=None, input_size=None, separator='_', test_nmb=50):
    if not os.path.exists(datadir):
        print("ERROR!The folder is not exist")
    # listdir = os.listdir(datadir)
    data = defaultdict(dict)
    image_dir = os.path.join(datadir, "imgs")
    image_paths = os.listdir(image_dir)
    for image_path in image_paths:
        nmb = image_path.split(separator)[0]
        data[nmb]['image'] = image_path
    anno_dir = os.path.join(datadir, "maps_bordered")
    anno_paths = os.listdir(anno_dir)
    for anno_path in anno_paths:
        nmb = anno_path.split(separator)[0]
        data[nmb]['anno'] = anno_path
    values = data.values()
    random.shuffle(values)
    return generate(values[test_nmb:], nb_classes, batch_size, input_size, image_dir, anno_dir), \
           generate(values[:test_nmb], nb_classes, batch_size, input_size, image_dir, anno_dir)


def update_inputs(batch_size=None, input_size=None, num_classes=None):
    return np.zeros([batch_size, input_size[0], input_size[1], 3]), \
           np.zeros([batch_size, input_size[0], input_size[1], num_classes])


def generate(values, nb_classes, batch_size, input_size, image_dir, anno_dir):
    while 1:
        random.shuffle(values)
        images, labels = update_inputs(batch_size=batch_size,
                                       input_size=input_size, num_classes=nb_classes)
        for i, d in enumerate(values):
            img = imresize(imread(os.path.join(image_dir, d['image']), mode='RGB'), input_size)
            y = imread(os.path.join(anno_dir, d['anno']), mode='L')
            h, w = input_size
            y = zoom(y, (1. * h / y.shape[0], 1. * w / y.shape[1]), order=1, prefilter=False)
            y = (np.arange(nb_classes) == y[:, :, None]).astype('float32')
            assert y.shape[2] == nb_classes
            images[i % batch_size] = img
            labels[i % batch_size] = y
            if (i + 1) % batch_size == 0:
                yield images, labels
                images, labels = update_inputs(batch_size=batch_size,
                                               input_size=input_size, num_classes=nb_classes)


def get_bounding_box_from_tiff(path):
    """
    Returns the four corners of the bounding box surrounding the TIF

    :param path:
    :return minx, miny, maxx, maxy:

    """

    # Retrieve the geo transform object from the tiff
    ds = gdal.Open(path)
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()

    # Calculate the corners of the bounding box
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]

    return minx, miny, maxx, maxy


def get_raster_size(path):
    """
    Get the pixel size of the raster at the path
    :param path:
    :return: width, height
    """
    ds = gdal.Open(path)
    width = ds.RasterXSize
    height = ds.RasterYSize
    return width, height


def is_tiff(name):
    """
    Returns if a file name is a tif or not
    :param name:
    :return:

    """
    return name.split('.')[-1] == 'tif'


def get_file_paths(path):
    """
    List and retrieve all the tif files in a directory
    :param path:
    :return: list of tif files
    """
    tiff_paths = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            # Check if the file is a tif
            if is_tiff(name):
                tiff_paths.append("{}/{}".format(path, name))
    return tiff_paths


def print_process(i, tot):
    print("Processing files {}%: 8{}D".format(int((i / tot) * 100), '=' * int(i / 2)), end="\r", flush=True)


def save_blank_raster(path, xsize, ysize):
    """
    Generate a blank raster with GDAL in the gives size.
    :param path:
    :param xsize:
    :param ysize:
    :return: None
    """
    file_format = "GTiff"
    driver = gdal.GetDriverByName(file_format)
    driver.Create(path, xsize=xsize, ysize=ysize,
                  bands=1, eType=gdal.GDT_Byte)


def save_file(content, path):
    """
    Save the content to the specified path.
    :param content:
    :param path:
    :return: None
    """
    file = open(path, 'wb')
    file.write(content)
    file.close()


def save_json(content, path):
    with open(path, 'w') as outfile:
        json.dump(content, outfile)


def make_path(path):
    """
    Create the requested path if it does not exist
    :param path:
    :return If the path was created:
    """
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

        # Returns true if the path was created
        return True

    # Returns false if the path was not created
    return False
