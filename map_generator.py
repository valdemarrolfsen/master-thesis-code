import argparse
import cv2
import os
import numpy as np

from keras_utils.generators import set_up_generators, load_images_from_folder
from keras_utils.smooth_tiled_predictions import predict_img_with_smooth_windowing
from networks.pspnet.net_builder import build_pspnet
from networks.unet.unet import build_unet


def image_to_neural_input(image_batch, image_datagen):

    generator = image_datagen.flow(
        image_batch,
        batch_size=image_batch.shape[0],
        shuffle=False
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
        0: [237, 237, 237],  # Empty
        1: [254, 241, 179],  # Roads
        2: [116, 173, 209],  # Water
        3: [193, 235, 176],  # Grass
        4: [170, 170, 170]  # Buildings
    }

    args = parser.parse_args()

    n_classes = args.classes
    model_name = args.model_name
    images_path = args.test_images
    input_size = args.input_size
    sample_path = os.path.join(args.sample_path, "examples")

    model_choices = {
        'pspnet': build_pspnet,
        'unet': build_unet
    }

    model_choice = model_choices[model_name]

    model = model_choice(n_classes, (input_size, input_size))

    model.load_weights(args.weights_path)

    # Set up the generators
    image_datagen, _ = set_up_generators(sample_path, rescale=False)

    window_size = input_size

    images = load_images_from_folder(images_path, num_samples=10000000)

    for i, input_img in enumerate(images):

        pred = predict_img_with_smooth_windowing(
            input_img,
            window_size=window_size,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=n_classes,
            pred_func=(
                lambda img_batch_subdiv: model.predict(
                    image_to_neural_input(img_batch_subdiv, image_datagen), verbose=True
                )
            )
        )

        print(pred.shape)

        result = np.argmax(pred, axis=2)

        seg_img = np.zeros((pred.shape[0], pred.shape[1], 3))

        for c in range(n_classes):
            seg_img[:, :, 0] += ((result[:, :] == c) * (class_color_map[c][2])).astype('uint8')
            seg_img[:, :, 1] += ((result[:, :] == c) * (class_color_map[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((result[:, :] == c) * (class_color_map[c][0])).astype('uint8')

        mask_name = "pred-{}.tif".format(i)
        img_name = "img-{}.tif".format(i)

        cv2.imwrite("{}/{}".format(args.output_path, mask_name), seg_img)
        cv2.imdecode("{}/{}".format(args.output_path, img_name), input_img)


if __name__ == '__main__':
    run()

