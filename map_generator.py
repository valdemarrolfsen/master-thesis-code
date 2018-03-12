import argparse
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras_utils.smooth_tiled_predictions import predict_img_with_smooth_windowing
from networks.pspnet.net_builder import build_pspnet
from networks.unet.unet import build_unet


def image_to_neural_input(image_batch):
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
    image_datagen.fit(image_batch)

    generator = image_datagen.flow(
        x=image_batch,
        batch_size=image_batch.shape[0],
        shuffle=False,
        seed=1
    )

    images = next(generator)

    return images


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=str)
    parser.add_argument("--epoch-number", type=int, default=5)
    parser.add_argument("--test-images", type=str, default="")
    parser.add_argument("--sample-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--input-size", type=int, default=713)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--classes", type=int)

    class_color_map = {
        0: [237, 237, 237],     # Empty
        1: [254, 241, 179],     # Roads
        2: [116, 173, 209],     # Water
        3: [193, 235, 176],     # Grass
        4: [170, 170, 170]      # Buildings
    }

    args = parser.parse_args()

    n_classes = args.classes
    model_name = args.model_name
    images_path = args.test_images
    input_size = args.input_size

    model_choices = {
        'pspnet': build_pspnet,
        'unet': build_unet
    }

    model_choice = model_choices[model_name]
    model = model_choice(n_classes, (input_size, input_size))
    model.load_weights(args.weights_path)

    images = [cv2.imread(images_path)]

    for i, input_img in enumerate(images):

        pred = predict_img_with_smooth_windowing(
            input_img,
            window_size=input_size,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=n_classes,
            pred_func=(
                lambda img_batch_subdiv: model.predict(
                    image_to_neural_input(img_batch_subdiv), verbose=True
                )
            )
        )

        result = np.argmax(pred, axis=2)

        seg_img = np.zeros((pred.shape[0], pred.shape[1], 3))

        for c in range(n_classes):
            seg_img[:, :, 0] += ((result[:, :] == c) * (class_color_map[c][2])).astype('uint8')
            seg_img[:, :, 1] += ((result[:, :] == c) * (class_color_map[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((result[:, :] == c) * (class_color_map[c][0])).astype('uint8')

        mask_name = "pred-{}.tif".format(i)
        img_name = "img-{}.tif".format(i)

        cv2.imwrite("{}/{}".format(args.output_path, mask_name), seg_img)
        cv2.imwrite("{}/{}".format(args.output_path, img_name), input_img)


if __name__ == '__main__':
    run()

