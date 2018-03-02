import argparse
import cv2
import numpy as np
import random

from networks.pspnet.net_builder import build_pspnet
from networks.unet.unet import build_unet
from keras_utils.generators import create_generator

parser = argparse.ArgumentParser()
parser.add_argument("--save-weights_path", type=str)
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

model = model_choice(n_classes, input_height=input_size, input_width=input_size, nChannels=3)

model.load_weights(args.save_weights_path)

generator = create_generator(images_path, (input_size, input_size), batch_size, n_classes)
images, masks = next(generator)

# Random colors for visualization
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

for i, img in enumerate(images):
    pr = model.predict(np.array([img]))[0]
    pr = pr.reshape((input_size, input_size, n_classes)).argmax(axis=2)
    seg_img = np.zeros((input_size, input_size, 3))

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    cv2.imwrite("test{}.tif".format(i), seg_img)
