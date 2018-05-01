import argparse
import cv2
import numpy as np
from tqdm import tqdm
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras_utils.generators import load_images_from_folder
from keras_utils.losses import binary_soft_jaccard_loss
from keras_utils.metrics import binary_jaccard_distance_rounded
from keras_utils.smooth_tiled_predictions import predict_img_with_smooth_windowing
from networks.pspnet.net_builder import build_pspnet
from networks.unet.unet import build_unet


def get_generator(images):
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

    image_datagen = ImageDataGenerator(**datagen_args)
    image_datagen.fit(images)
    return image_datagen


def image_to_neural_input(images, image_datagen):
    generator = image_datagen.flow(
        x=images,
        batch_size=images.shape[0],
        shuffle=False,
    )
    images = next(generator)
    return images


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=str)
    parser.add_argument("--test-images", type=str, default="")
    parser.add_argument("--sample-images", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--input-size", type=int, default=713)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--classes", type=int)

    args = parser.parse_args()
    n_classes = args.classes
    model_name = args.model_name
    images_path = args.test_images
    sample_path = args.sample_images
    input_size = args.input_size

    model_choices = {
        'pspnet': build_pspnet,
        'unet': build_unet
    }

    model_choice = model_choices[model_name]
    model = model_choice((input_size, input_size), 1)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=binary_soft_jaccard_loss,
        metrics=['acc', binary_jaccard_distance_rounded])
    model.load_weights(args.weights_path)

    sample_images = np.array(load_images_from_folder(sample_path, num_samples=500))
    generator = get_generator(sample_images)
    sample_images = None
    # load all images
    images = load_images_from_folder(images_path, num_samples=100000)

    pred = predict_img_with_smooth_windowing(
        images[0],
        window_size=input_size,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=n_classes,
        pred_func=(
            lambda img_batch_subdiv: model.predict(
                image_to_neural_input(img_batch_subdiv, generator), verbose=True
            )
        )
    )

    print(pred)


if __name__ == '__main__':
    run()
