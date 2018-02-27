import json

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

    def get_tif_from_bbox(self, min_x, max_y, max_x, min_y, table_name, color_attribute='255'):
        x_res = 1000
        y_res = 1000
        x_scale = (max_x - min_x) / x_res
        y_scale = (max_y - min_y) / y_res
        query = """
        WITH mygeoms AS (
          SELECT st_asraster(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)),
            ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
            '8BUI', {color_attribute}::INTEGER, 0) as rast
          FROM {table_name}
          WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
        ),
        empty as (
          SELECT st_asraster(
                  st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833), 
                  ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
                '8BUI', 0, 0) as rast
        )
        SELECT ST_AsGDALRaster(st_union(foo.rast, 'sum'),'GTiff')
        FROM (SELECT rast FROM mygeoms UNION SELECT rast FROM empty) foo
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

    def get_geojson_from_bbox(self, min_x, max_y, max_x, min_y, table_name):
        query = """
            SELECT jsonb_build_object(
                'type',     'FeatureCollection',
                'features', jsonb_agg(feature)
            )
            FROM (
              SELECT jsonb_build_object(
                'type',       'Feature',
                'id',         gid,
                'geometry',   ST_AsGeoJSON(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)))::jsonb,
                'properties', to_jsonb(row) - 'gid' - 'geom'
              ) AS feature
              FROM (SELECT * FROM {table_name}) as row WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))) features;
        """.format(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            table_name=table_name)

        if not self.cursor:
            raise ValueError("No cursor detected! Is the current generator connected to a database?")

        self.cursor.execute(query)
        records = self.cursor.fetchall()
        return records