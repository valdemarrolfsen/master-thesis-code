import psycopg2

database_server = "129.241.91.231"
database_name = "rubval"
database_user = "postgres"
database_password = "postgres"


class Db(object):
    """
    The generator class is responsible for creating a data set
    """

    def __init__(self):
        self.conn = None
        self.cursor = None

    def connect(self):
        connect_string = "host={} dbname={} user={} password={}".format(
            database_server,
            database_name,
            database_user,
            database_password
        )

        self.conn = psycopg2.connect(connect_string)
        self.cursor = self.conn.cursor()

        # GDAL hack
        self.cursor.execute("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';")
        self.cursor.execute("SET postgis.enable_outdb_rasters TO True;")

    def get_tif_from_bbox(self, min_x, min_y, max_x, max_y, table_name, color_attribute='255'):
        x_res = 1000
        y_res = 1000
        x_scale = (max_x - min_x) / 1000
        y_scale = (max_y - min_y) / 1000
        query = """
        WITH mygeoms AS (
          SELECT st_asraster(geom,
            ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
            ARRAY['8BUI', '8BUI', '8BUI'], ARRAY[{color_attribute}::INTEGER,{color_attribute}::INTEGER,{color_attribute}::INTEGER], ARRAY[0,0,0]) as rast
          FROM {table_name}
          WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
        )
        SELECT ST_AsGDALRaster(ST_Clip(st_union(rast), st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833), TRUE),'GTiff')
        FROM mygeoms
        """.format(
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                    x_res=x_res,
                    y_res=y_res,
                    x_scale=x_scale,
                    y_scale=y_scale,
                    table_name=table_name,
                    color_attribute=color_attribute
                )

        if not self.cursor:
            raise ValueError("No cursor detected! Is the current generator connected to a database?")

        self.cursor.execute(query)
        records = self.cursor.fetchall()

        return records

