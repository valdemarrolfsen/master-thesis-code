from generator.generator import TiffGenerator
from osgeo import ogr

# Test file path
from generator.utils import get_bounding_box_from_tiff

file_path = "data/trondheim_small"
output_path = "data/output"


def run():

    # Create a new tiff generator
    generator = TiffGenerator(file_path)

    # Connect to the postgres database
    generator.connect_to_db()

    tiff_files = generator.get_file_paths()

    for file in tiff_files:
        min_x, min_y, max_x, max_y = get_bounding_box_from_tiff(file)

        records = generator.get_geometry_from_bounding_box(
            min_x,
            min_y,
            max_x,
            max_y
        )

        print(len(records))

        for rec in records:
            b = bytes(rec[0])
            g = ogr.CreateGeometryFromWkb(b)
            if g is not None:
                print(g.ExportToWkt())


if __name__ == '__main__':
    run()
