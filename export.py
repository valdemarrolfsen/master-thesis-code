import utils
import cv2
import threading
from functools import reduce
from queue import Queue, Empty


def generate_set(input_path, output_path, size=200, layover=0.1, input_size=1000):
    """
    Generates a training set by loading all examples into memory, and resizing them.

    :param input_path:
    :param output_path:
    :param size:
    :param layover:
    :param input_size:
    :return:
    """

    # Assuming that the files are located in the folders 'labels' and 'examples'
    label_paths = utils.get_file_paths("{}/labels".format(input_path))
    example_paths = utils.get_file_paths("{}/examples".format(input_path))

    # Defines the output path based on the size
    output_path = "{0}/{1}x{1}".format(output_path, size)

    export_path_example = "{}/examples/".format(output_path)
    export_path_label = "{}/labels/".format(output_path)

    # Make the path if it does not exist
    utils.make_path(export_path_example)
    utils.make_path(export_path_label)

    path_length = len(label_paths)

    q = Queue()
    for i in range(path_length):
        q.put(i)

    for i in range(10):
        # Create a new database connection for each thread.
        t = threading.Thread(
            target=work,
            args=(
                q,
                example_paths,
                label_paths,
                path_length,
                export_path_example,
                export_path_label,
                size,
                layover,
                input_size
            )
        )

        # Sticks the thread in a list so that it remains accessible
        t.daemon = True
        t.start()

    q.join()
    print("")


def mask_image(image_path, export_path, size, layover=0.5, input_size=1000):
    """
    Generates a set of images of the specified size
    with a layover as equal as possible to the specified layover.

    :param image_path:
    :param export_path:
    :param size:
    :param layover:
    :param input_size:
    :return:
    """

    # Loads the image and make sure that all images have the same size
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_size, input_size))

    # Calculating a valid stride size
    stride_size = (1 - layover) * size
    sliding_space = image.shape[0] - size
    possible_factors = factors(sliding_space)
    stride_size = min(possible_factors, key=lambda factor_number: abs(factor_number - stride_size))

    iterations = int(sliding_space / stride_size)

    name = image_path.split('/')[-1].split('.')[0]
    img_format = image_path.split('/')[-1].split('.')[-1]

    for i in range(iterations):
        y = i * stride_size
        for j in range(iterations):
            x = j * stride_size
            crop_img = image[y:y + size, x:x + size]
            path = "{}{}-{}_{}.{}".format(export_path, name, i, j, img_format)
            cv2.imwrite(path, crop_img)


def factors(n):
    """
    Returns all factors for input parameter n.

    :param n:
    :return:
    """
    f = list(reduce(list.__add__, ([i, n // i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0)))
    return sorted(f)


def work(q, example_paths, label_paths, total_files, export_path_example, export_path_label, size, layover, input_size):
    """
    The worker that produces the masks for a single pair of examples and labels

    :param q:
    :param example_paths:
    :param label_paths:
    :param total_files:
    :param export_path_example:
    :param export_path_label:
    :param size:
    :param layover:
    :param input_size:
    :return:
    """

    while not q.empty():
        try:
            i = q.get(False)
        except Empty:
            break

        # Show progress
        utils.print_process(total_files - q.qsize(), total_files)

        # We assume that related examples and labels have the same index in the path lists
        example_path = example_paths[i]
        label_path = label_paths[i]

        # Creates masks for the image pairs
        mask_image(example_path, export_path_example, size, layover, input_size)
        mask_image(label_path, export_path_label, size, layover, input_size)

        q.task_done()


if __name__ == "__main__":
    generate_set('data/output', 'data/export')
