import cv2
import math
from skimage import io

from osgeo import osr
from osgeo import gdal

maps_zoom_scales = {
    20: 1128.497220,
    19: 2256.994440,
    18: 4513.988880,
    17: 9027.977761,
    16: 18055.955520,
    15: 36111.911040,
    14: 72223.822090,
    13: 144447.644200,
    12: 288895.288400,
    11: 577790.576700,
    10: 1155581.153000,
    9: 2311162.307000,
    8: 4622324.614000,
    7: 9244649.227000,
    6: 18489298.450000,
    5: 36978596.910000,
    4: 73957193.820000,
    3: 147914387.600000,
    2: 295828775.300000,
    1: 591657550.500000
}

bounding_box = [
    63.4375248,  # ul_lat
    10.3443029,  # ul_lng

    63.4059196,  # dr_lat
    10.4510333   # dr_lng
]


def geo_reference_raster(raster_path, lat, lon, size):
    """

    :param raster_path:
    :return:
    """

    src_ds = gdal.Open(raster_path)
    format = "GTiff"
    driver = gdal.GetDriverByName(format)

    # Open destination dataset
    dst_ds = driver.CreateCopy(raster_path, src_ds, 0)

    # Make WGS84 lon lat coordinate system
    world_sr = osr.SpatialReference()
    world_sr.SetWellKnownGeogCS('WGS84')

    # Get raster projection
    epsg = 3857
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dest_wkt = srs.ExportToWkt()

    coord_transform = osr.CoordinateTransformation(world_sr, srs)
    newpoints = coord_transform.TransformPoints([[lat, lon]])

    # Specify raster location through geotransform array
    # (uperleftx, scalex, skewx, uperlefty, skewy, scaley)
    gt = [newpoints[0][0], size, 0, newpoints[0][1], 0, -size]

    # Set location
    dst_ds.SetGeoTransform(gt)

    # Set projection
    dst_ds.SetProjection(dest_wkt)

    # Close files
    dst_ds = None
    src_ds = None


def slide_window(lat, lng, size, zoom):
    parallel_multiplier = math.cos(lat * math.pi / 180)
    degrees_per_pixelx = 360 / math.pow(2, zoom + 8)
    degrees_per_pixely = 360 / math.pow(2, zoom + 8) * parallel_multiplier

    point_lat = lat - degrees_per_pixely * size
    point_lng = lng + degrees_per_pixelx * size

    return point_lat, point_lng


def run():
    lat = bounding_box[0]
    lng = bounding_box[1]

    zoom = 16
    size = 400
    map_type = "satellite"
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    api_key = "AIzaSyD3sIrrRRqyFNKrHeW58bplkmqHUXuG_Hg"

    i = 0
    j = 0

    while lat > bounding_box[2]:
        while lng < bounding_box[3]:
            url = "{base_url}?center={lat},{lng}&zoom={zoom}&size={width}x{height}&maptype={type}&key={key}".format(
                base_url=base_url,
                lat=lat,
                lng=lng,
                zoom=zoom,
                width=size,
                height=size+40,  # So that we can crop out the google logo
                type=map_type,
                key=api_key
            )

            img = io.imread(url)
            img = img[0:size, 0:size]

            save_path = "tiles/tile_{}_{}.tif".format(i, j)

            cv2.imwrite(save_path, img)

            geo_reference_raster(save_path, lat, lng, 1)

            _, lng = slide_window(lat, lng, size, zoom)

            j += 1

        # Jump to the next row
        lng = bounding_box[1]
        lat, _ = slide_window(lat, lng, size, zoom)

        print("Finished with row {}".format(i))

        j = 0
        i += 1


if __name__ == '__main__':
    run()
