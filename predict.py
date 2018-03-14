import argparse
import cv2
import os
import numpy as np

from osgeo import ogr, osr
from osgeo import gdal

from networks.pspnet.net_builder import build_pspnet
from networks.unet.unet import build_unet
from keras_utils.generators import create_generator


def get_real_image(path, name, use_gdal=False):
    """
    Returns a raster image with the geo frame intact

    :param path:
    :param name:
    :return:
    """

    image_path = os.path.join(path, 'examples', name)

    if use_gdal:
        return gdal.Open(image_path)

    return cv2.imread(image_path)


def get_geo_frame(raster):
    """
    Retrieves the coordinates of a geo referenced raster

    :param raster:
    :return:
    """
    ulx, scalex, skewx, uly, skewy, scaley = raster.GetGeoTransform()

    return ulx, scalex, skewx, uly, skewy, scaley


def geo_reference_raster(raster_path, coordinates):
    """

    :param raster_path:
    :param coordinates:
    :return:
    """

    src_ds = gdal.Open(raster_path)
    format = "GTiff"
    driver = gdal.GetDriverByName(format)

    # Open destination dataset
    dst_ds = driver.CreateCopy("{}_referenced.tif".format(os.path.splitext(raster_path)[0]), src_ds, 0)

    # Specify raster location through geotransform array
    # (uperleftx, scalex, skewx, uperlefty, skewy, scaley)
    gt = coordinates

    # Set location
    dst_ds.SetGeoTransform(gt)

    # Get raster projection
    epsg = 3857
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dest_wkt = srs.ExportToWkt()

    # Set projection
    dst_ds.SetProjection(dest_wkt)

    # Close files
    dst_ds = None
    src_ds = None


parser = argparse.ArgumentParser()
parser.add_argument("--weights-path", type=str)
parser.add_argument("--epoch-number", type=int, default=5)
parser.add_argument("--test-images", type=str, default="")
parser.add_argument("--output-path", type=str, default="")
parser.add_argument("--input-size", type=int, default=713)
parser.add_argument("--batch-size", type=int, default=713)
parser.add_argument("--model-name", type=str, default="")
parser.add_argument("--classes", type=int)


class_color_map = {
    0: [237, 237, 237],  # Empty
    1: [254, 241, 179],  # Roads
    2: [116, 173, 209],  # Water
    3: [193, 235, 176],  # Grass
    4: [170, 170, 170]  # Buildings
}

args = parser.parse_args()

n_classes = args.classes
model_name = args.model_name
images_path = args.test_images
input_size = args.input_size
batch_size = args.batch_size

model_choices = {
    'pspnet': build_pspnet,
    'unet': build_unet
}

model_choice = model_choices[model_name]

model = model_choice(n_classes, (input_size, input_size))

model.load_weights(args.weights_path)

generator, _ = create_generator(
    images_path,
    (input_size, input_size),
    batch_size,
    n_classes,
    rescale=False,
    with_file_names=True
)

images, masks, file_names = next(generator)
probs = model.predict(images, verbose=1)

for i, prob in enumerate(probs):
    result = np.argmax(prob, axis=2)
    mask_result = np.argmax(masks[i], axis=2)
    img = get_real_image(images_path, file_names[i])
    raster = get_real_image(images_path, file_names[i], use_gdal=True)

    seg_img = np.zeros((input_size, input_size, 3))
    seg_mask = np.zeros((input_size, input_size, 3))

    for c in range(n_classes):
        seg_img[:, :, 0] += ((result[:, :] == c) * (class_color_map[c][2])).astype('uint8')
        seg_img[:, :, 1] += ((result[:, :] == c) * (class_color_map[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((result[:, :] == c) * (class_color_map[c][0])).astype('uint8')

        seg_mask[:, :, 0] += ((mask_result[:, :] == c) * (class_color_map[c][2])).astype('uint8')
        seg_mask[:, :, 1] += ((mask_result[:, :] == c) * (class_color_map[c][1])).astype('uint8')
        seg_mask[:, :, 2] += ((mask_result[:, :] == c) * (class_color_map[c][0])).astype('uint8')

    pred_name = "pred-{}.tif".format(i)
    pred_save_path = "{}/{}".format(args.output_path, pred_name)

    cv2.imwrite(pred_save_path, seg_img)
    cv2.imwrite("{}/mask-{}.tif".format(args.output_path, i), seg_mask)
    cv2.imwrite("{}/image-{}.tif".format(args.output_path, i), img)

    # Get coordinates for corresponding image
    ulx, scalex, skewx, uly, skewy, scaley = get_geo_frame(raster)

    # Geo reference newly created raster
    geo_reference_raster(
        pred_save_path,
        [ulx, scalex, skewx, uly, skewy, scaley]
    )
