import argparse
import os
import cv2
import numpy as np
from keras import backend as K
from keras.optimizers import Adam

from keras_utils.generators import create_generator
from keras_utils.losses import binary_soft_jaccard_loss
from keras_utils.metrics import batch_general_jaccard, f1_score, binary_jaccard_distance_rounded
from keras_utils.multigpu import get_number_of_gpus, ModelMGPU
from networks.densenet.densenet import build_densenet
from networks.unet.unet import build_unet

class_color_map = {
    0: [237, 237, 237],  # Empty
    1: [254, 241, 179],  # Roads
    2: [116, 173, 209],  # Water
    3: [193, 235, 176],  # Grass
    4: [170, 170, 170]  # Buildings
}

datasets = [
    {'name': 'buildings', 'size': 2570},
    {'name': 'roads', 'size': 2830},
    {'name': 'water', 'size': 1352},
    {'name': 'vegetation', 'size': 2952},
]

scores = {
    'buildings': 4,
    'roads': 1,
    'water': 2,
    'vegetation': 3
}

models = [
    {'name': 'unet', 'input_size': 512, 'method': build_unet},
    {'name': 'densenet', 'input_size': 320, 'method': build_densenet}]


def pred():
    image_path = '/data/{}/test'
    for dataset in datasets:
        im_path = image_path.format(dataset['name'])

        for model in models:
            input_size = model['input_size']
            generator, _ = create_generator(
                im_path,
                (input_size, input_size),
                dataset['size'],
                1,
                rescale=True,
                with_file_names=True,
                binary=True,
                mean=np.array([[[0.36654497, 0.35386439, 0.30782658]]]),
                std=np.array([[[0.19212837, 0.19031791, 0.18903286]]])
            )
            images, masks, file_names = next(generator)

            m = model['method']((input_size, input_size), 1)
            gpus = get_number_of_gpus()
            if gpus > 1:
                m = ModelMGPU(m, gpus)

            m.compile(
                optimizer=Adam(lr=1e-4),
                loss=binary_soft_jaccard_loss,
                metrics=['acc', binary_jaccard_distance_rounded])

            weights_path = 'weights_train/weights.{}-{}-final-finetune.h5'.format(model['name'], dataset['name'])
            m.load_weights(weights_path)

            probs = m.predict(images, verbose=1)
            probs = np.round(probs)
            iou = batch_general_jaccard(masks, probs)
            f1 = f1_score(masks, probs)
            print('Mean IOU for {} on {}: {}'.format(model['name'], dataset['name'], np.mean(iou)))
            print('F1 score for {} on {}: {}'.format(model['name'], dataset['name'], f1))

            # wow such hack
            from keras_utils.prediction import get_real_image, get_geo_frame, geo_reference_raster

            for i, (prob, mask) in enumerate(zip(probs, masks)):
                if i > 200:
                    break
                raster = get_real_image(im_path, file_names[i], use_gdal=True)
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

                out_path = 'finalpreds/{}/{}'.format(model['name'], dataset['name'])
                pred_save_path = "{}/{}".format(out_path, pred_name)

                cv2.imwrite(pred_save_path, prob)
                cv2.imwrite("{}/image-{}.tif".format(out_path, i), img)
                cv2.imwrite("{}/mask-{}.tif".format(out_path, i), mask)

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
    all_probs = {}

    model_choices = {
        'densenet': build_densenet,
        'unet': build_unet
    }

    model_choice = model_choices[model_name]

    generator, _ = create_generator(
        images_path,
        (input_size, input_size),
        batch_size,
        5,
        rescale=True,
        with_file_names=True,
        binary=False,
        mean=np.array([[[0.36654497, 0.35386439, 0.30782658]]]),
        std=np.array([[[0.19212837, 0.19031791, 0.18903286]]])
    )
    images, masks, file_names = next(generator)

    for dataset in datasets:
        model = model_choice((input_size, input_size), 1)

        gpus = get_number_of_gpus()
        print('Fund {} gpus'.format(gpus))
        if gpus > 1:
            model = ModelMGPU(model, gpus)

        model.compile(
            optimizer=Adam(lr=1e-4),
            loss=binary_soft_jaccard_loss,
            metrics=['acc', binary_jaccard_distance_rounded])

        model.load_weights(args.weights_path.format(dataset))

        probs = model.predict(images, verbose=1)
        probs = np.round(probs)
        # iou = batch_general_jaccard(masks, probs, binary=True)
        # f1 = K.eval(f1_score(K.variable(masks), K.variable(probs)))
        # print('Mean IOU for {}: {}'.format(dataset, np.mean(iou)))
        # print('F1 score for {}: {}'.format(dataset, f1))

        all_probs[dataset] = probs

    final_prob = None

    for i, key in enumerate(all_probs):
        prob = all_probs[key]
        prob[prob == 1] = scores[key]

        if i == 0:  # First iteration
            final_prob = prob
            continue

        final_prob = np.maximum.reduce([final_prob, prob])

    masks = np.argmax(masks, axis=2)
    iou = batch_general_jaccard(masks, final_prob)
    f1 = K.eval(f1_score(K.variable(masks), K.variable(final_prob)))
    print('Mean IOU for {}: {}'.format('multiclass', np.mean(iou)))
    print('F1 score for {}: {}'.format('multiclass', f1))

    if not save_imgs:
        return

    # wow such hack
    from keras_utils.prediction import get_real_image, get_geo_frame, geo_reference_raster

    for i, prob in enumerate(final_prob):
        mask = np.argmax(masks[i], axis=2)
        raster = get_real_image(os.path.join(images_path, 'multiclass', 'test'), file_names[i], use_gdal=True)
        R = raster.GetRasterBand(1).ReadAsArray()
        G = raster.GetRasterBand(2).ReadAsArray()
        B = raster.GetRasterBand(3).ReadAsArray()
        img = np.zeros((512, 512, 3))
        img[:, :, 0] = B
        img[:, :, 1] = G
        img[:, :, 2] = R

        seg_pred = np.zeros((input_size, input_size, 3))
        seg_mask = np.zeros((input_size, input_size, 3))

        for c in range(5):
            seg_pred[:, :, 0] += ((prob[:, :, 0] == c) * (class_color_map[c][2])).astype('uint8')
            seg_pred[:, :, 1] += ((prob[:, :, 0] == c) * (class_color_map[c][1])).astype('uint8')
            seg_pred[:, :, 2] += ((prob[:, :, 0] == c) * (class_color_map[c][0])).astype('uint8')

            seg_mask[:, :, 0] += ((mask[:, :] == c) * (class_color_map[c][2])).astype('uint8')
            seg_mask[:, :, 1] += ((mask[:, :] == c) * (class_color_map[c][1])).astype('uint8')
            seg_mask[:, :, 2] += ((mask[:, :] == c) * (class_color_map[c][0])).astype('uint8')

        pred_name = "pred-{}.tif".format(i)
        pred_save_path = "{}/{}".format(args.output_path, pred_name)

        cv2.imwrite(pred_save_path, seg_pred)
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


if __name__ == '__main__':
    pred()
