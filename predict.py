import argparse
import cv2
import os
import numpy as np

import shapely.wkt
import shapely.affinity
from osgeo import ogr
from shapely.geometry import MultiPolygon, Polygon

from collections import defaultdict

from networks.pspnet.net_builder import build_pspnet
from networks.unet.unet import build_unet
from keras_utils.generators import create_generator


def simplify_contours(contours, epsilon):
    return [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]


def find_child_parent(hierarchy, approx_contours, min_area):
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1

    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            holes = [c[:, 0, :] for c in cnt_children.get(idx, []) if cv2.contourArea(c) >= min_area]
            contour = cnt[:, 0, :]

            poly = Polygon(shell=contour, holes=holes)

            if poly.area >= min_area:
                all_polygons.append(poly)

    return all_polygons


def fix_invalid_polygons(all_polygons):
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def mask2polygons_layer(mask, epsilon=1.0, min_area=10.0):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(mask.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    # create approximate contours to have reasonable submission size
    if epsilon != 0:
        approx_contours = simplify_contours(contours, epsilon)
    else:
        approx_contours = contours

    if not approx_contours:
        return MultiPolygon()

    all_polygons = find_child_parent(hierarchy, approx_contours, min_area)

    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)

    all_polygons = fix_invalid_polygons(all_polygons)

    return all_polygons


def mask2poly(predicted_mask, x_scaler, y_scaler):
    min_area = 10

    polygons = mask2polygons_layer(predicted_mask, epsilon=0, min_area=min_area)

    polygons = shapely.affinity.scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))

    return polygons


def fix_raster(path, name):
    gdal_siev = "gdal_sieve.py -st {0} {1} {2} ".format(
        5,
        "{}/{}".format(path, name),
        "{}/fix_{}".format(path, name)
    )

    os.system(gdal_siev)


def save_to_shp(collection, i):
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource('shape_{}.shp'.format(i))
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)

    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(collection.wkb)
    feat.SetGeometry(geom)

    layer.CreateFeature(feat)
    feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None


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
    0: [0, 0, 0],  # Empty
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

generator, _ = create_generator(images_path, (input_size, input_size), batch_size, n_classes)
images, masks = next(generator)

probs = model.predict(images, verbose=1)

for i, prob in enumerate(probs):
    result = np.argmax(prob, axis=2)
    mask_result = np.argmax(masks[i], axis=2)
    img = images[i]
    img = (img*255).astype('uint8')
    seg_img = np.zeros((input_size, input_size, 3))
    seg_mask = np.zeros((input_size, input_size, 3))

    poly_collection = mask2poly(result, 1, 1)
    poly_collection = poly_collection.simplify(0.05, preserve_topology=False)

    save_to_shp(poly_collection, i)

    for c in range(n_classes):
        seg_img[:, :, 0] += ((result[:, :] == c) * (class_color_map[c][2])).astype('uint8')
        seg_img[:, :, 1] += ((result[:, :] == c) * (class_color_map[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((result[:, :] == c) * (class_color_map[c][0])).astype('uint8')

        seg_mask[:, :, 0] += ((mask_result[:, :] == c) * (class_color_map[c][2])).astype('uint8')
        seg_mask[:, :, 1] += ((mask_result[:, :] == c) * (class_color_map[c][1])).astype('uint8')
        seg_mask[:, :, 2] += ((mask_result[:, :] == c) * (class_color_map[c][0])).astype('uint8')

    mask_name = "pred-{}.tif".format(i)

    cv2.imwrite("{}/{}".format(args.output_path, mask_name), seg_img)
    cv2.imwrite("{}/mask-{}.tif".format(args.output_path, i), seg_mask)
    cv2.imwrite("{}/image-{}.tif".format(args.output_path, i), img)
