import threading
from queue import Queue, Empty
import utils
from db import Db
file_path = "data/trondheim_small"
output_path = "data/output/"


def run():
    global file_path
    tiff_files = utils.get_file_paths(file_path)
    table_name = 'veg_flate'
    color_attribute = '255'
    total_files = len(tiff_files)
    print('Using table {}'.format(table_name))
    print('Found {} files'.format(total_files))

    q = Queue()
    for file in tiff_files:
        q.put(file)

    for i in range(10):
        # Create a new database connection for each thread.
        db = Db()
        db.connect()
        t = threading.Thread(target=work, args=(q, db, table_name, color_attribute))
        # Sticks the thread in a list so that it remains accessible
        t.daemon = True
        t.start()

    q.join()


def work(q, db, table_name, color_attribute):
    while not q.empty():
        try:
            file = q.get(False)
        except Empty:
            break

        min_x, min_y, max_x, max_y = utils.get_bounding_box_from_tiff(file)
        records = db.get_tif_from_bbox(
            min_x,
            min_y,
            max_x,
            max_y,
            table_name,
            color_attribute
        )

        filename = 'label_' + file.split('/')[-1]
        path = output_path + filename
        width, height = utils.get_raster_size(file)
        if not records:
            utils.save_blank_raster(path, width, height)
            q.task_done()
            continue
        rast = records[0][0]
        if rast is None:
            utils.save_blank_raster(path, width, height)
        else:
            utils.save_file(rast, path)
        q.task_done()


if __name__ == '__main__':
    run()
