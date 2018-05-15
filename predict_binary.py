import argparse

import cv2
import numpy as np
from keras import backend as K
from keras.optimizers import Adam

from keras_utils.generators import create_generator
from keras_utils.losses import binary_soft_jaccard_loss
from keras_utils.metrics import batch_general_jaccard, f1_score, binary_jaccard_distance_rounded, maximize_threshold
from networks.densenet.densenet import build_densenet
from networks.unet.unet import build_unet, build_unet_old


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=str)
    parser.add_argument("--epoch-number", type=int, default=5)
    parser.add_argument("--test-images", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--input-size", type=int, default=713)
    parser.add_argument("--batch-size", type=int, default=713)
    parser.add_argument("--model-name", type=str, default="standard")
    parser.add_argument("--save-imgs", type=bool, default=True)
    args = parser.parse_args()

    model_name = args.model_name
    images_path = args.test_images
    input_size = args.input_size
    batch_size = args.batch_size
    save_imgs = args.save_imgs

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
    generator, _ = create_generator(
        images_path,
        (input_size, input_size),
        batch_size,
        1,
        rescale=True,
        with_file_names=True,
        binary=True,
        mean=[0.01279744, 0.01279744, 0.01279744],
        std=[0.11312577, 0.11312577, 0.11312577]
    )

    images, masks, file_names = next(generator)
    probs = model.predict(images, verbose=1)
    iou = batch_general_jaccard(masks, probs, binary=True)
    f1 = K.eval(f1_score(K.variable(masks), K.variable(probs)))
    print('mean IOU: {}'.format(np.mean(iou)))
    print('F1 score: {}'.format(f1))

    maximize_threshold(masks, probs)

    if not save_imgs:
        return
    # wow such hack
    from keras_utils.prediction import get_real_image, get_geo_frame, geo_reference_raster
    for i, prob in enumerate(probs):
        # mask_result = np.argmax(masks[i], axis=2)
        # img = get_real_image(images_path, file_names[i])
        mask = masks[i]
        raster = get_real_image(images_path, file_names[i], use_gdal=True)
        R = raster.GetRasterBand(1).ReadAsArray()
        G = raster.GetRasterBand(2).ReadAsArray()
        B = raster.GetRasterBand(3).ReadAsArray()
        img = np.zeros((512, 512, 3))
        img[:, :, 0] = B
        img[:, :, 1] = G
        img[:, :, 2] = R
        prob = np.round(prob)
        prob = (prob[:, :, 0] * 255.).astype(np.uint8)
        mask = (mask[:, :, 0] * 255.).astype(np.uint8)
        pred_name = "pred-{}.tif".format(i)
        pred_save_path = "{}/{}".format(args.output_path, pred_name)

        cv2.imwrite(pred_save_path, prob)
        cv2.imwrite("{}/image-{}.tif".format(args.output_path, i), img)
        cv2.imwrite("{}/mask-{}.tif".format(args.output_path, i), mask)

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
