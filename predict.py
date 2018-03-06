import argparse
import cv2
import numpy as np

from networks.pspnet.net_builder import build_pspnet
from networks.unet.unet import build_unet
from keras_utils.generators import create_generator

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
    0: [44, 62, 80],  # Empty
    1: [204, 142, 53],  # Buildings
    2: [165, 177, 194],  # Roads
    3: [52, 172, 224],  # Water
    4: [38, 222, 129],  # Grass
    5: [0, 148, 50],  # Forest
    6: [60, 99, 130],  # Developed
}

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

generator, _ = create_generator(images_path, (input_size, input_size), batch_size, n_classes)
images, masks = next(generator)

probs = model.predict(images, verbose=1)

for i, prob in enumerate(probs):
    result = np.argmax(prob, axis=2)
    print(np.unique(result))
    img = images[i]
    img = (img*255).astype('uint8')
    seg_img = np.zeros((input_size, input_size, 3))

    for c in range(n_classes):
        seg_img[:, :, 0] += ((result[:, :] == c) * (class_color_map[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((result[:, :] == c) * (class_color_map[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((result[:, :] == c) * (class_color_map[c][2])).astype('uint8')

    cv2.imwrite("{}/pred-{}.tif".format(args.output_path, i), seg_img)
    cv2.imwrite("{}/image-{}.tif".format(args.output_path, i), img)
