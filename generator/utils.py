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
    miny = gt[3] + width*gt[4] + height*gt[5]
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]

    return minx, miny, maxx, maxy


def is_tiff(name):
    """
    Returns if a file name is a tif or not
    :param name:
    :return:

    """
    return name.split('.')[-1] == 'tif'
