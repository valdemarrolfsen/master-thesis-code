import errno
import json
import os

from keras.preprocessing.image import ImageDataGenerator
from osgeo import gdal


def create_generator(datadir=''):
    image_dir = os.path.join(datadir, "examples")
    label_dir = os.path.join(datadir, "labels")

    datagen_args = dict(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0,
        # randomly shift images vertically
        height_shift_range=0,
        # randomly flip images
        horizontal_flip=False,
        # randomly flip images
        vertical_flip=False)

    image_datagen = ImageDataGenerator(**datagen_args)
    label_datagen = ImageDataGenerator(**datagen_args)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # Use the same seed for both generators so they return corresponding images
    seed = 1

    # image_datagen.fit(images, augment=True, seed=seed)
    # label_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
        image_dir,
        target_size=(713, 713),
        class_mode=None,
        seed=seed)

    label_generator = label_datagen.flow_from_directory(
        label_dir,
        target_size=(713, 713),
        class_mode=None,
        seed=seed)

    generator = zip(image_generator, label_generator)
    return generator


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
    if height != width:
        print('not square image ', path)
        return -1, -1, -1, -1
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
    print("Processing files {}%: 8{}D".format(int((i / tot) * 100), '=' * int((i / tot) * 100)), end="\r", flush=True)


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
