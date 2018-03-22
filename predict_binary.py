import argparse

import cv2
import numpy as np

from keras_utils.generators import create_generator
from keras_utils.prediction import get_real_image, get_geo_frame, geo_reference_raster
from networks.unet_binary.unet import build_unet_binary_deeper_elu


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=str)
    parser.add_argument("--epoch-number", type=int, default=5)
    parser.add_argument("--test-images", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--input-size", type=int, default=713)
    parser.add_argument("--batch-size", type=int, default=713)
    parser.add_argument("--model-name", type=str, default="")
    args = parser.parse_args()

    model_name = args.model_name
    images_path = args.test_images
    input_size = args.input_size
    batch_size = args.batch_size

    model_choices = {
        'unet': build_unet_binary_deeper_elu,
    }

    model_choice = model_choices[model_name]
    model = model_choice((input_size, input_size))
    model.load_weights(args.weights_path)

    generator, _ = create_generator(
        images_path,
        (input_size, input_size),
        batch_size,
        2,
        rescale=False,
        flip=False,
        with_file_names=True,
        binary=True
    )

    images, masks, file_names = next(generator)
    probs = model.predict(images, verbose=1)

    for i, prob in enumerate(probs):
        # mask_result = np.argmax(masks[i], axis=2)
        # img = get_real_image(images_path, file_names[i])
        raster = get_real_image(images_path, file_names[i], use_gdal=True)
        img = np.array(raster.GetRasterBand(1).ReadAsArray())

        prob = (prob[:, :, 0] * 255.).astype(np.uint8)
        pred_name = "pred-{}.tif".format(i)
        pred_save_path = "{}/{}".format(args.output_path, pred_name)

        cv2.imwrite(pred_save_path, prob)
        cv2.imwrite("{}/image-{}.tif".format(args.output_path, i), img)

        try:
            # Get coordinates for corresponding image
            ulx, scalex, skewx, uly, skewy, scaley = get_geo_frame(raster)

            # Geo reference newly created raster
            geo_reference_raster(
                pred_save_path,
                [ulx, scalex, skewx, uly, skewy, scaley]
            )
        except ValueError as e:
            print("Was not able to reference image at path: {}".format(pred_save_path))


if __name__ == '__main__':
    run()
