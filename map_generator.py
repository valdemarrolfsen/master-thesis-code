import argparse
import cv2
import os

from keras_utils.generators import set_up_generators, load_images_from_folder
from keras_utils.smooth_tiled_predictions import predict_img_with_smooth_windowing
from networks.pspnet.net_builder import build_pspnet
from networks.unet.unet import build_unet


def image_to_neural_input(image_batch, image_datagen, input_size):
    generator = image_datagen.fit(
        image_batch,
        target_size=input_size,
        class_mode=None,
        shuffle=False,
    )

    return next(generator)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=str)
    parser.add_argument("--epoch-number", type=int, default=5)
    parser.add_argument("--test-images", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--input-size", type=int, default=713)
    parser.add_argument("--batch-size", type=int, default=713)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--classes", type=int)

    args = parser.parse_args()

    n_classes = args.classes
    model_name = args.model_name
    images_path = args.test_images
    input_size = args.input_size
    batch_size = args.batch_size

    model_choices = {
        'pspnet': build_pspnet,
        'unet': build_unet
    }

    model_choice = model_choices[model_name]

    model = model_choice(n_classes, (input_size, input_size))

    model.load_weights(args.weights_path)

    image_dir = os.path.join(images_path, 'examples')

    # Set up the generators
    image_datagen, _ = set_up_generators(image_dir, rescale=False)

    window_size = batch_size

    images = load_images_from_folder(images_path, num_samples=10000000)

    for i, input_img in enumerate(images):

        predictions_smooth = predict_img_with_smooth_windowing(
            input_img,
            window_size=window_size,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=n_classes,
            pred_func=(
                lambda img_batch_subdiv: model.predict(image_to_neural_input(img_batch_subdiv, image_datagen))
            )
        )

        print(predictions_smooth)

        cv2.imwrite("{}/pred-{}.tif".format(args.output_path, i), predictions_smooth)


if __name__ == '__main__':
    run()
