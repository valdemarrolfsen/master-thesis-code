import os

from keras_utils.generators import set_up_generators


def create_caps_generator(datadir, input_size, batch_size, nb_classes, rescale=False, augment=False, with_file_names=False, binary=False):
    image_dir = os.path.join(datadir, "examples")
    label_dir = os.path.join(datadir, "labels")

    # Set up the generators
    image_datagen, label_datagen = set_up_generators(image_dir, rescale)

    # Use the same seed for both generators so they return corresponding images
    seed = 1

    shuffle = True

    if with_file_names:
        shuffle = False

    image_generator = image_datagen.flow_from_directory(
        image_dir,
        batch_size=batch_size,
        target_size=input_size,
        class_mode=None,
        shuffle=shuffle,
        seed=seed)

    label_generator = label_datagen.flow_from_directory(
        label_dir,
        batch_size=batch_size,
        target_size=input_size,
        class_mode=None,
        shuffle=shuffle,
        color_mode='grayscale',
        seed=seed)

    generator = zip(image_generator, label_generator)

    if augment and not binary:
        raise NotImplementedError('Augment for categorical not implemented. Implement it in the generator.')

    return binary_gen(generator, batch_size), image_generator.samples


def binary_gen(generator, batch_size):
    """
    Generator that cleans data and returns binary labels.
    """
    while True:
        img, mask = next(generator)

        if len(img) != batch_size:
            continue

        # clean mask image
        mask[mask > 1] = 0

        yield ([img, mask], [mask, mask * img])
