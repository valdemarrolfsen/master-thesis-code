import gc

import cv2
import os
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras_utils.augment import image_augmentation


def load_images_from_folder(folder, num_samples=5000):
    images = []
    for filename in os.listdir(folder):
        fold = os.path.join(folder, filename)
        if os.path.isdir(fold):
            folder = fold
            break

    for i, filename in enumerate(tqdm(os.listdir(folder))):
        if i > num_samples:
            break
        imgpath = os.path.join(folder, filename)
        if not os.path.isfile(imgpath):
            continue
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)
    return images


def set_up_generators(image_dir, rescale, fit=True):
    datagen_args = dict(
        data_format='channels_last',
        # set input mean to 0 over the dataset
        featurewise_center=True,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=True,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0,
        # randomly shift images vertically
        height_shift_range=0,
        # randomly flip images
        horizontal_flip=False,
        # randomly flip images
        vertical_flip=False)

    if rescale:
        # Scale down the values
        datagen_args['rescale'] = 1. / 255

    image_datagen = ImageDataGenerator(**datagen_args)

    # We do not want to augment the labels other than skew and shift
    datagen_args['rescale'] = None
    datagen_args['featurewise_std_normalization'] = False
    datagen_args['featurewise_center'] = False
    label_datagen = ImageDataGenerator(**datagen_args)

    if not fit:
        return image_datagen, label_datagen
    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    imgs = np.array(load_images_from_folder(image_dir))

    if len(imgs) < 1:
        raise ValueError('No images found in {}'.format(image_dir))

    if rescale:
        imgs = imgs.astype(np.float32) / 255

    image_datagen.fit(imgs)
    del imgs
    gc.collect()
    print('Train mean and std')
    print(image_datagen.mean)
    print(image_datagen.std)

    return image_datagen, label_datagen


def create_generator(datadir, input_size, batch_size, nb_classes, rescale=False, augment=False, with_file_names=False, binary=False, mean=None,
                     std=None):
    image_dir = os.path.join(datadir, "examples")
    label_dir = os.path.join(datadir, "labels")

    if mean is not None and std is not None:
        fit = False
    else:
        fit = True
    # Set up the generators
    image_datagen, label_datagen = set_up_generators(image_dir, rescale, fit)

    if not fit:
        image_datagen.mean = mean
        image_datagen.std = std
        label_datagen.mean = mean
        label_datagen.std = std

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

    file_name_generator = None
    if with_file_names:
        file_name_generator = image_generator

    if augment and not binary:
        raise NotImplementedError('Augment for categorical not implemented. Implement it in the generator.')

    # If we are doing binary predictions, we do not want to one-hot encode the labels.
    if binary:
        return custom_binary_gen(generator, batch_size, file_name_generator, augment), image_generator.samples

    return custom_gen(
        generator,
        input_size,
        batch_size,
        nb_classes,
        file_name_generator
    ), image_generator.samples


def custom_gen(generator, input_size, batch_size, nb_classes, file_name_generator):
    """
    Generator that cleans data and returns one-hot encoded labels for multiclass prediction.
    :param generator:
    :param input_size:
    :param batch_size:
    :param nb_classes:
    :param file_name_generator:
    :return:
    """
    while True:
        img, mask = next(generator)

        if len(img) != batch_size:
            continue

        mask[mask > nb_classes - 1] = 0
        output = np.ndarray((batch_size, input_size[0], input_size[1], nb_classes))
        for i in range(mask.shape[0]):
            output[i] = to_categorical(mask[i], num_classes=nb_classes)

        if file_name_generator:
            idx = (file_name_generator.batch_index - 1) * file_name_generator.batch_size
            file_names = file_name_generator.filenames[idx: idx + file_name_generator.batch_size]
            yield img, output, file_names
        else:
            yield img, output


def custom_binary_gen(generator, batch_size, file_name_generator, augment):
    """
    Generator that cleans data and returns binary labels.
    :param generator:
    :param batch_size:
    :param file_name_generator:
    :return:
    """
    while True:
        img, mask = next(generator)

        if len(img) != batch_size:
            continue

        # clean mask image
        mask[mask > 1] = 0

        if augment:
            # Augment the images
            img, mask = image_augmentation(img, mask)

        if file_name_generator:
            idx = (file_name_generator.batch_index - 1) * file_name_generator.batch_size
            file_names = file_name_generator.filenames[idx: idx + file_name_generator.batch_size]
            yield img, mask, file_names
        else:
            yield img, mask
