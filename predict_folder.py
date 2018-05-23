import argparse

import cv2
import numpy as np
import os
from PIL import Image
from keras.optimizers import Adam
from tqdm import tqdm

from keras_utils.losses import binary_soft_jaccard_loss
from keras_utils.metrics import binary_jaccard_distance_rounded
from keras_utils.multigpu import get_number_of_gpus, ModelMGPU
from networks.densenet.densenet import build_densenet
from networks.unet.unet import build_unet, build_unet_old


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=str)
    parser.add_argument("--input-images", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--input-size", type=int, default=713)
    parser.add_argument("--model-name", type=str, default="standard")
    parser.add_argument("--save-imgs", type=bool, default=True)
    args = parser.parse_args()

    model_name = args.model_name
    images_path = args.input_images
    input_size = args.input_size
    save_imgs = args.save_imgs

    model_choices = {
        'densenet': build_densenet,
        'unet': build_unet,
        'unet-old': build_unet_old
    }

    model_choice = model_choices[model_name]
    model = model_choice((input_size, input_size), 1)

    gpus = get_number_of_gpus()
    if gpus > 1:
        model = ModelMGPU(model, gpus)

    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=binary_soft_jaccard_loss,
        metrics=['acc', binary_jaccard_distance_rounded])

    model.load_weights(args.weights_path)

    probs = []
    for i, filename in enumerate(tqdm(os.listdir(images_path))):
        imgpath = os.path.join(images_path, filename)
        if not os.path.isfile(imgpath):
            continue
        img = np.array(Image.open(imgpath))
        if not img.shape[0] == img.shape[1]:
            continue
        img = np.expand_dims(img, axis=0)
        prob = model.predict(img, verbose=1)
        probs.append(prob)
    probs = np.round(probs)
    if not save_imgs:
        return
    for i, prob in enumerate(probs):
        prob = (prob[:, :, 0] * 255.).astype(np.uint8)
        pred_name = "pred-{}.tif".format(i)
        pred_save_path = "{}/{}".format(args.output_path, pred_name)
        cv2.imwrite(pred_save_path, prob)


if __name__ == '__main__':
    run()
