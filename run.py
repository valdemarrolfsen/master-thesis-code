from generator.generator import TiffGenerator
from osgeo import ogr, gdal

# Test file path
from generator.utils import get_bounding_box_from_tiff

file_path = "data/trondheim_small"
output_path = "data/output/"


def run():
    # Create a new tiff generator
    generator = TiffGenerator(file_path)

    # Connect to the postgres database
    generator.connect_to_db()

    tiff_files = generator.get_file_paths()

    table_name = 'ar5_flate'
    color_attribute = 'artype'
    total_files = len(tiff_files)
    print('Using table {}'.format(table_name))
    print('Found {} files'.format(total_files))
    for i, file in enumerate(tiff_files):
        print_process(i, total_files)
        min_x, min_y, max_x, max_y = get_bounding_box_from_tiff(file)
        records = generator.get_tif_from_bbox(
            min_x,
            min_y,
            max_x,
            max_y,
            table_name,
            color_attribute
        )
        rast = records[0][0]
        filename = 'label_' + file.split('/')[-1]
        path = output_path + filename

        if rast is None:
            save_blank_raster(path)
        else:
            save_raster(rast, path)


def print_process(i, tot):
    print("Processing file {}%: 8{}D".format(int((i / tot) * 100), '=' * int(i / 2)), end="\r", flush=True)


def save_blank_raster(path, xsize, ysize):
    file_format = "GTiff"
    driver = gdal.GetDriverByName(file_format)
    driver.Create(path, xsize=xsize, ysize=ysize,
                  bands=1, eType=gdal.GDT_Byte)


def save_raster(rast, path):
    file = open(path, 'wb')
    file.write(rast)
    file.close()


if __name__ == '__main__':
    run()
