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

        print("Connected!")

    def get_geometry_from_bounding_box(self, min_x, min_y, max_x, max_y):
        query = """
                    SELECT st_AsBinary(geom)
                    FROM veg_flate
                    WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
                """.format(
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y
                )

        if not self.cursor:
            raise ValueError("No cursor detected! Is the current generator connected to a database?")

        self.cursor.execute(query)
        records = self.cursor.fetchall()

        return records

