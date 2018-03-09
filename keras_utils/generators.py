import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


def load_images_from_folder(folder, num_samples=5000):
    images = []
    for filename in os.listdir(folder):
        fold = os.path.join(folder, filename)
        if os.path.isdir(fold):
            folder = fold
            break

    for i, filename in enumerate(os.listdir(folder)):
        if i > num_samples:
            break
        imgpath = os.path.join(folder, filename)
        if not os.path.isfile(imgpath):
            continue
        img = cv2.imread(imgpath)
        if img is not None:
            images.append(img)
    return images


def create_generator(datadir, input_size, batch_size, nb_classes, rescale=False, with_file_names=False):
    image_dir = os.path.join(datadir, "examples")
    label_dir = os.path.join(datadir, "labels")

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

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    imgs = load_images_from_folder(image_dir)
    image_datagen.fit(imgs)

    # Use the same seed for both generators so they return corresponding images
    seed = 1

    image_generator = image_datagen.flow_from_directory(
        image_dir,
        batch_size=batch_size,
        target_size=input_size,
        class_mode=None,
        seed=seed)

    label_generator = label_datagen.flow_from_directory(
        label_dir,
        batch_size=batch_size,
        target_size=input_size,
        class_mode=None,
        color_mode='grayscale',
        seed=seed)

    generator = zip(image_generator, label_generator)

    file_names = None

    if with_file_names:
        file_names = image_generator.filenames

    return custom_gen(
        generator,
        input_size,
        batch_size,
        nb_classes,
        file_names
    ), image_generator.samples


def custom_gen(generator, input_size, batch_size, nb_classes, file_names):
    while True:
        img, mask = next(generator)

        if len(img) != batch_size:
            continue

        mask[mask > nb_classes-1] = 0
        output = np.ndarray((batch_size, input_size[0], input_size[1], nb_classes))
        for i in range(mask.shape[0]):
            output[i] = to_categorical(mask[i], num_classes=nb_classes)

        if file_names:
            idx = (generator.batch_index - 1) * generator.batch_size
            file_names = file_names[idx: idx + generator.batch_size]
            yield img, output, file_names
        else:
            yield img, output
