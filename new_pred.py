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
        rescale=False,
        with_file_names=True,
        binary=True
    )

    images, masks, file_names = next(generator)
    probs = model.predict(images, verbose=1)
    iou = batch_general_jaccard(masks, probs, binary=True)
    f1 = K.eval(f1_score(K.variable(masks), K.variable(probs)))
    print('mean IOU: {}'.format(np.mean(iou)))
    print('F1 score: {}'.format(f1))

    if not save_imgs:
        return
    for i, prob in enumerate(probs):
        # mask_result = np.argmax(masks[i], axis=2)
        # img = get_real_image(images_path, file_names[i])
        mask = masks[i]
        img = images[i]
        prob = np.round(prob)
        prob = (prob[:, :, 0] * 255.).astype(np.uint8)
        mask = (mask[:, :, 0] * 255.).astype(np.uint8)
        pred_name = "pred-{}.tif".format(i)
        pred_save_path = "{}/{}".format(args.output_path, pred_name)

        cv2.imwrite(pred_save_path, prob)
        cv2.imwrite("{}/image-{}.tif".format(args.output_path, i), img)
        cv2.imwrite("{}/mask-{}.tif".format(args.output_path, i), mask)


if __name__ == '__main__':
    run()
