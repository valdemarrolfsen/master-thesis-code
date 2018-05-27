import argparse
import cv2
import numpy as np
from keras.optimizers import Adam

from keras_utils.generators import create_generator
from keras_utils.losses import binary_soft_jaccard_loss
from keras_utils.metrics import batch_general_jaccard, f1_score, binary_jaccard_distance_rounded, batch_classwise_general_jaccard, \
    batch_classwise_f1_score
from keras_utils.multigpu import get_number_of_gpus, ModelMGPU
from keras_utils.prediction import get_real_image, get_geo_frame, geo_reference_raster
from networks.densenet.densenet import build_densenet
from networks.unet.unet import build_unet
from networks.pspnet.net_builder import build_pspnet


def run(args):
    n_classes = args.classes
    model_name = args.model_name
    images_path = args.test_images
    input_size = args.input_size
    batch_size = args.batch_size
    save_imgs = args.save_imgs

    model_choices = {
        'unet': build_unet,
        'densenet': build_densenet,
        'pspnet': build_pspnet
    }

    model_choice = model_choices[model_name]

    model = model_choice((input_size, input_size), n_classes)

    gpus = get_number_of_gpus()
    print('Fund {} gpus'.format(gpus))
    if gpus > 1:
        model = ModelMGPU(model, gpus)

    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=binary_soft_jaccard_loss,
        metrics=['acc', binary_jaccard_distance_rounded])

    model.load_weights(args.weights_path)

    binary = n_classes == 1
    generator, samples = create_generator(
        images_path,
        (input_size, input_size),
        batch_size,
        nb_classes=n_classes,
        rescale_masks=False,
        with_file_names=True,
        binary=binary,
        augment=False,
        mean=np.array([[[0.36654497, 0.35386439, 0.30782658]]]),
        std=np.array([[[0.19212837, 0.19031791, 0.18903286]]])
    )

    steps = samples // batch_size
    f1s = []
    ious = []
    for i in range(steps):
        images, masks, file_names = next(generator)
        probs = model.predict(images, verbose=1)

        probs = np.argmax(probs, axis=3)
        masks = np.argmax(masks, axis=3)
        iou = batch_classwise_general_jaccard(masks, probs)
        f1 = batch_classwise_f1_score(masks, probs)
        ious.append(iou)
        f1s.append(f1)

        if not save_imgs:
            continue

        for i, prob in enumerate(probs):
            result = prob
            mask_result = masks[i]

            # img = get_real_image(images_path, file_names[i])
            raster = get_real_image(images_path, file_names[i], use_gdal=True)
            R = raster.GetRasterBand(1).ReadAsArray()
            G = raster.GetRasterBand(2).ReadAsArray()
            B = raster.GetRasterBand(3).ReadAsArray()
            img = np.zeros((512, 512, 3))
            img[:, :, 0] = B
            img[:, :, 1] = G
            img[:, :, 2] = R

            seg_img = np.zeros((input_size, input_size, 3))
            seg_mask = np.zeros((input_size, input_size, 3))

            for c in range(n_classes):
                seg_img[:, :, 0] += ((result[:, :] == c) * (class_color_map[c][2])).astype('uint8')
                seg_img[:, :, 1] += ((result[:, :] == c) * (class_color_map[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((result[:, :] == c) * (class_color_map[c][0])).astype('uint8')

                seg_mask[:, :, 0] += ((mask_result[:, :] == c) * (class_color_map[c][2])).astype('uint8')
                seg_mask[:, :, 1] += ((mask_result[:, :] == c) * (class_color_map[c][1])).astype('uint8')
                seg_mask[:, :, 2] += ((mask_result[:, :] == c) * (class_color_map[c][0])).astype('uint8')

            pred_name = "pred-{}.tif".format(i)
            pred_save_path = "{}/{}".format(args.output_path, pred_name)

            cv2.imwrite(pred_save_path, seg_img)
            cv2.imwrite("{}/mask-{}.tif".format(args.output_path, i), seg_mask)
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

    print('Mean IOU: {}'.format(np.mean(ious, axis=0)))
    print('F1 score: {}'.format(np.mean(f1s)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=str)
    parser.add_argument("--test-images", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--input-size", type=int, default=713)
    parser.add_argument("--batch-size", type=int, default=713)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--classes", type=int)
    parser.add_argument("--save-imgs", type=bool, default=True)

    class_color_map = {
        0: [237, 237, 237],  # Empty
        1: [254, 241, 179],  # Roads
        2: [116, 173, 209],  # Water
        3: [193, 235, 176],  # Grass
        4: [170, 170, 170]  # Buildings
    }

    args = parser.parse_args()
    run(args)
