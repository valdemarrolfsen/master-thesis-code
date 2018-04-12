import argparse

import cv2
import numpy as np

from keras_utils.generators import create_generator
from keras_utils.prediction import get_real_image, get_geo_frame, geo_reference_raster
from networks.densenet.densenet import build_densenet
from networks.unet.unet import build_unet

parser = argparse.ArgumentParser()
parser.add_argument("--weights-path", type=str)
parser.add_argument("--epoch-number", type=int, default=5)
parser.add_argument("--test-images", type=str, default="")
parser.add_argument("--output-path", type=str, default="")
parser.add_argument("--input-size", type=int, default=713)
parser.add_argument("--batch-size", type=int, default=713)
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
batch_size = args.batch_size

model_choices = {
    'unet': build_unet,
    'densenet': build_densenet
}

model_choice = model_choices[model_name]

model = model_choice((input_size, input_size), n_classes)

model.load_weights(args.weights_path)

generator, _ = create_generator(
    images_path,
    (input_size, input_size),
    batch_size,
    n_classes,
    rescale=False,
    with_file_names=True
)

images, masks, file_names = next(generator)
probs = model.predict(images, verbose=1)

for i, prob in enumerate(probs):
    result = np.argmax(prob, axis=2)
    mask_result = np.argmax(masks[i], axis=2)
    # img = get_real_image(images_path, file_names[i])
    raster = get_real_image(images_path, file_names[i], use_gdal=True)
    R = raster.GetRasterBand(1).ReadAsArray()
    G = raster.GetRasterBand(2).ReadAsArray()
    B = raster.GetRasterBand(3).ReadAsArray()
    img = np.zeros((input_size, input_size, 3))
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
