import cv2
import math
from skimage import io

from osgeo import osr
from osgeo import gdal

bounding_box = [
    63.4375248,  # ul_lat
    10.3443029,  # ul_lng

    63.4059196,  # dr_lat
    10.4510333  # dr_lng
]


def eccentricity(a, b):
    return math.sqrt(1 - (b / a) ** 2)


def n_rad(a, b, lat):
    ec = eccentricity(a, b)
    return a / (1 - (ec ** 2) * math.sin(lat) ** 2) ** (1 / 2)


def degree_to_meter(a, b, lat, lon, h):
    N = n_rad(a, b, lat)

    x = (N + h) * math.cos(lat) * math.cos(lon)
    y = (N + h) * math.cos(lat) * math.sin(lon)
    z = (N * (b / a) ** 2 + h) * math.sin(lat)

    return x, y, z


def georeference_raster(raster_path, lat, lon):
    """

    :param lon: Longitude
    :param lat: Latitude
    :param scale: Image scale
    :param raster_path: Path to raster image
    :return: None
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
    epsg = 4326
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dest_wkt = srs.ExportToWkt()

    # coord_transform = osr.CoordinateTransformation(world_sr, srs)
    # newpoints = coord_transform.TransformPoints([[lon, lat]])

    min_y, min_x = point_from_pixels(lat, lon, 16, -200)
    max_y, max_x = point_from_pixels(lat, lon, 16, 200)

    x_scale = (max_x - min_x) / 400
    y_scale = (max_y - min_y) / 400

    # Specify raster location through geotransform array
    # (x_min, pixel_size, 0, y_max, 0, -pixel_size)
    gt = [min_x, x_scale, 0, max_y, 0, -y_scale]

    # Set location
    dst_ds.SetGeoTransform(gt)

    # Set projection
    dst_ds.SetProjection(dest_wkt)

    # Close files
    dst_ds = None
    src_ds = None


def get_google_maps_scale(lat, zoom):
    """
    Get the meters per pixel scale of the lat and zoom level
    :param lat:
    :param zoom:
    :return: scale
    """
    return 156543.03392 * math.cos(lat * math.pi / 180) / math.pow(2, zoom)


def slide_window(lat, lng, size, zoom):
    parallel_multiplier = math.cos(lat * math.pi / 180)
    degrees_per_pixelx = 360 / math.pow(2, zoom + 8)
    degrees_per_pixely = 360 / math.pow(2, zoom + 8) * parallel_multiplier

    point_lat = lat - degrees_per_pixely * size
    point_lng = lng + degrees_per_pixelx * size

    return point_lat, point_lng


def point_from_pixels(lat, lng, zoom, pixels):
    parallel_multiplier = math.cos(lat * math.pi / 180)
    degrees_per_pixelx = 360 / math.pow(2, zoom + 8)
    degrees_per_pixely = 360 / math.pow(2, zoom + 8) * parallel_multiplier

    point_lat = lat + degrees_per_pixely * pixels
    point_lng = lng + degrees_per_pixelx * pixels
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
                height=size + 40,  # So that we can crop out the google logo
                type=map_type,
                key=api_key
            )

            img = io.imread(url)
            img = img[0:size, 0:size]

            save_path = "tiles/tile_{}_{}.tif".format(i, j)

            cv2.imwrite(save_path, img)

            # temp_x, temp_y, _ = degree_to_meter(a, b, lat, lng, h)
            georeference_raster(save_path, lat, lng)

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
