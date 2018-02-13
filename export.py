from osgeo import gdal


def generate_set(path):
    gtif = gdal.Open(path)
    print(gtif.GetMetadata())
