import os
import psycopg2

from .utils import is_tiff

database_server = "129.241.91.231"
database_name = "rubval"
database_user = "postgres"
database_password = "postgres"


class TiffGenerator(object):
    """
    The generator class is responsible for creating a data set
    """

    def __init__(self, tiff_path):
        self.tiff_path = tiff_path
        self.conn = None
        self.cursor = None

    def get_file_paths(self):
        tiff_paths = []

        for root, dirs, files in os.walk(self.tiff_path, topdown=False):
            for name in files:
                # Check if the file is a tif
                if is_tiff(name):
                    tiff_paths.append("{}/{}".format(self.tiff_path, name))

        return tiff_paths

    def connect_to_db(self):
        connect_string = "host={} dbname={} user={} password={}".format(
            database_server,
            database_name,
            database_user,
            database_password
        )

        print("Connecting to database: {}".format(database_name))

        self.conn = psycopg2.connect(connect_string)
        self.cursor = self.conn.cursor()
        # GDAL hack
        self.cursor.execute("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';")
        self.cursor.execute("SET postgis.enable_outdb_rasters TO True;")

        print("Connected!")

    def get_tif_from_bbox(self, min_x, min_y, max_x, max_y, table_name, color_attribute='255'):
        query = """
        WITH mygeoms AS (
          SELECT st_asraster(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)),
            ST_MakeEmptyRaster( 1000, 1000, 1, 1, 0.5, 0.5, 0, 0, 25833),
            ARRAY['8BUI', '8BUI', '8BUI'], ARRAY[{color_attribute}::INTEGER,{color_attribute}::INTEGER,{color_attribute}::INTEGER], ARRAY[0,0,0]) as rast
          FROM {table_name}
          WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
        )
        SELECT ST_AsGDALRaster(st_union(rast),'GTiff')
        FROM mygeoms
        """.format(
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                    table_name=table_name,
                    color_attribute=color_attribute
                )

        if not self.cursor:
            raise ValueError("No cursor detected! Is the current generator connected to a database?")

        self.cursor.execute(query)
        records = self.cursor.fetchall()

        return records

