import os
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


def extract_imagelist(image_dir, input_size):
    image_dir = image_dir + '/buildings'

    image_files = []

    for entry in os.listdir(image_dir):
        file_path = "{}/{}".format(image_dir, entry)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.tif')):
            image_files.append(file_path)

    image_len = len(image_files)
    all_images = np.empty([image_len, input_size, input_size, 3])

    for i in range(image_len):
        img = image.load_img(image_files[i], target_size=(input_size, input_size))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        all_images[i, :, :, :] = img

    return all_images


def create_generator(datadir='', input_size=(713, 713), batch_size=32):
    image_dir = os.path.join(datadir, "examples")
    label_dir = os.path.join(datadir, "labels")

    datagen_args = dict(
        data_format='channels_last',
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
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

    image_datagen = ImageDataGenerator(**datagen_args)
    label_datagen = ImageDataGenerator(**datagen_args)

    images = extract_imagelist(image_dir, input_size[0])
    labels = extract_imagelist(label_dir, input_size[0])

    print(labels)

    labels = to_categorical(labels)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # Use the same seed for both generators so they return corresponding images
    seed = 1

    image_datagen.fit(images, augment=True, seed=seed)
    label_datagen.fit(labels, augment=True, seed=seed)

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
        seed=seed)

    generator = zip(image_generator, label_generator)
    return generator
