import argparse

import cv2
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras_utils.generators import load_images_from_folder
from keras_utils.losses import binary_soft_jaccard_loss
from keras_utils.metrics import binary_jaccard_distance_rounded
from networks.densenet.densenet import build_densenet
from networks.unet.unet import build_unet, build_unet_old


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
    generator = image_datagen.flow(
        x=images,
        batch_size=images.shape[0],
        shuffle=False,
    )
    return generator

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=str)
    parser.add_argument("--test-images", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--input-size", type=int, default=713)
    parser.add_argument("--model-name", type=str, default="standard")
    args = parser.parse_args()

    model_name = args.model_name
    images_path = args.test_images
    input_size = args.input_size

    model_choices = {
        'densenet': build_densenet,
        'unet': build_unet,
        'unet-old': build_unet_old
    }

    model_choice = model_choices[model_name]
    model = model_choice((input_size, input_size), 1)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=binary_soft_jaccard_loss,
        metrics=['acc', binary_jaccard_distance_rounded])
    model.load_weights(args.weights_path)

    images = load_images_from_folder(images_path)
    generator = get_generator(images)
    probs = model.predict_generator(generator, verbose=1)
    for i, prob in enumerate(probs):

        img = images[i]
        prob = np.round(prob)
        prob = (prob[:, :, 0] * 255.).astype(np.uint8)
        pred_name = "pred-{}.tif".format(i)
        pred_save_path = "{}/{}".format(args.output_path, pred_name)

        cv2.imwrite(pred_save_path, prob)
        cv2.imwrite("{}/image-{}.tif".format(args.output_path, i), img)


if __name__ == '__main__':
    run()
