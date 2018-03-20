from collections import defaultdict

import cv2
import os
import shapely.affinity
import shapely.wkt
from osgeo import gdal
from osgeo import ogr, osr
from shapely.geometry import MultiPolygon, Polygon


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


def geo_reference_raster(raster_path, geotransform):
    """

    :param raster_path:
    :param geotransform:
    :return:
    """

    src_ds = gdal.Open(raster_path)
    format = "GTiff"
    driver = gdal.GetDriverByName(format)

    # Open destination dataset
    dst_ds = driver.CreateCopy(raster_path, src_ds, 0)

    # Specify raster location through geotransform array
    # (uperleftx, scalex, skewx, uperlefty, skewy, scaley)
    gt = geotransform

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

