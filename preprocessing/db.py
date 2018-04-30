import json

import psycopg2

database_server = "129.241.91.231"
database_name = "rubval"
database_user = "postgres"
database_password = "rubval123"

CLASS_TABLE_MAPPING = {
    'buildings': ['bygning_flate', 'bygnanlegg_flate'],
    'roads': ['veg_flate']
}


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

    def get_tif_from_bbox(self, min_x, max_y, max_x, min_y, res, color_attribute='255'):
        x_res = res
        y_res = res
        x_scale = (max_x - min_x) / x_res
        y_scale = (max_y - min_y) / y_res
        query = """
        WITH area AS (
          SELECT st_asraster(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)),
            ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
            '8BUI', {color_attribute}::INTEGER, 0) as rast
          FROM ar5_flate
          WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
        ),
        areatype AS (
          SELECT st_asraster(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)),
            ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
            '8BUI', {color_attribute}::INTEGER, 0) as rast
          FROM arealbruk_flate
          WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
        ),
        roads AS (
          SELECT st_asraster(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)),
            ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
            '8BUI', {color_attribute}::INTEGER, 0) as rast
          FROM veg_flate
          WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
        ),
        buildings AS (
          SELECT st_asraster(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)),
            ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
            '8BUI', {color_attribute}::INTEGER, 0) as rast
          FROM bygning_flate
          WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
        ),
        structures AS (
          SELECT st_asraster(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)),
            ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
            '8BUI', {color_attribute}::INTEGER, 0) as rast
          FROM bygnanlegg_flate
          WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
        ),
        water AS (
          SELECT st_asraster(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)),
            ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
            '8BUI', {color_attribute}::INTEGER, 0) as rast
          FROM vann_flate
          WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
        ),
        empty as (
          SELECT st_asraster(
                  st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833), 
                  ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
                '8BUI', 0, 0) as rast
        )
        SELECT ST_AsGDALRaster(st_union(foo.rast, 'max'),'GTiff')
        FROM (
          SELECT rast FROM area
          UNION SELECT rast from areatype
          UNION SELECT rast from roads
          UNION SELECT rast from buildings
          UNION SELECT rast from structures
          UNION SELECT rast from water
          UNION SELECT rast from empty) foo
        """.format(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            x_res=x_res,
            y_res=y_res,
            x_scale=x_scale,
            y_scale=y_scale,
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

    def get_binary_tiff(self, min_x, max_y, max_x, min_y, res, class_name, color_attribute='255'):
        query = self._get_query(min_x, max_y, max_x, min_y, res, class_name, color_attribute)
        if not self.cursor:
            raise ValueError("No cursor detected! Is the current generator connected to a database?")

        self.cursor.execute(query)
        records = self.cursor.fetchall()
        return records

    @staticmethod
    def _get_query(min_x, max_y, max_x, min_y, res, class_name, color_attribute):
        x_res = res
        y_res = res
        x_scale = (max_x - min_x) / x_res
        y_scale = (max_y - min_y) / y_res

        if class_name == 'buildings':
            query = """
                WITH area AS (
                  SELECT st_asraster(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)),
                    ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
                    '8BUI', {color_attribute}::INTEGER, 0) as rast
                  FROM bygning_flate
                  WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
                ),
                structures AS (
                  SELECT st_asraster(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)),
                    ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
                    '8BUI', {color_attribute}::INTEGER, 0) as rast
                  FROM bygnanlegg_flate
                  WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
                ),
                empty as (
                  SELECT st_asraster(
                          st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833), 
                          ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
                        '8BUI', 0, 0) as rast
                )
                SELECT ST_AsGDALRaster(st_union(foo.rast, 'max'),'GTiff')
                FROM (
                  SELECT rast FROM area
                  UNION SELECT rast FROM structures
                  UNION SELECT rast FROM empty) foo
                """.format(
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
                x_res=x_res,
                y_res=y_res,
                x_scale=x_scale,
                y_scale=y_scale,
                color_attribute=color_attribute
            )
        elif class_name == 'roads':
            query = """
                    WITH area AS (
                      SELECT st_asraster(st_intersection(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833)),
                        ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
                        '8BUI', {color_attribute}::INTEGER, 0) as rast
                      FROM veg_flate
                      WHERE st_intersects(geom, st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833))
                    ),
                    empty as (
                      SELECT st_asraster(
                              st_makeenvelope({min_x}, {min_y}, {max_x}, {max_y}, 25833), 
                              ST_MakeEmptyRaster({x_res}, {y_res}, {min_x}::FLOAT, {max_y}::FLOAT, {x_scale}, {y_scale}, 0, 0, 25833),
                            '8BUI', 0, 0) as rast
                    )
                    SELECT ST_AsGDALRaster(st_union(foo.rast, 'max'),'GTiff')
                    FROM (
                      SELECT rast FROM area
                      UNION SELECT rast FROM empty) foo
                    """.format(
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
                x_res=x_res,
                y_res=y_res,
                x_scale=x_scale,
                y_scale=y_scale,
                color_attribute=color_attribute
            )
        else:
            raise ValueError('Unknown class type')
        return query
