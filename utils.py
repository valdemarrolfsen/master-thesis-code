import os

import errno
from osgeo import gdal


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
    i = tot - i
    print("Processing file {}%: 8{}D".format(int((i / tot) * 100), '=' * int(i / 2)), end="\r", flush=True)


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
