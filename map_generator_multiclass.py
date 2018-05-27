import os
import argparse
import cv2
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras_utils.losses import binary_soft_jaccard_loss
from keras_utils.metrics import binary_jaccard_distance_rounded, batch_general_jaccard, f1_score, general_jaccard, batch_classwise_general_jaccard
from keras_utils.multigpu import get_number_of_gpus, ModelMGPU
from keras_utils.smooth_tiled_predictions import predict_img_with_smooth_windowing, cheap_tiling_prediction, overlapping_predictions
from networks.densenet.densenet import build_densenet
from networks.unet.unet import build_unet


def get_generator():
    datagen_args = dict(
        data_format='channels_last',
        # set input mean to 0 over the dataset
        featurewise_center=True,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=True,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0,
        # randomly shift images vertically
        height_shift_range=0,
        # randomly flip images
        horizontal_flip=False,
        # randomly flip images
        vertical_flip=False,

        rescale=1. / 255)

    image_datagen = ImageDataGenerator(**datagen_args)
    image_datagen.mean = np.array([[[0.36654497, 0.35386439, 0.30782658]]])
    image_datagen.std = np.array([[[0.19212837, 0.19031791, 0.18903286]]])
    return image_datagen


def image_to_neural_input(images, image_datagen):
    generator = image_datagen.flow(
        x=images,
        batch_size=images.shape[0],
        shuffle=False,
    )
    images = next(generator)
    return images


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=str)
    parser.add_argument("--test-image", type=str, default="")
    parser.add_argument("--sample-images", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--input-size", type=int, default=713)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--mask", type=str, default="")

    args = parser.parse_args()
    model_name = args.model_name
    image_path = args.test_image
    input_size = args.input_size
    output_path = args.output_path
    file_name = args.name
    mask_path = args.mask

    nb_classes = 5
    model_choices = {
        'unet': build_unet,
        'densenet': build_densenet
    }

    model_choice = model_choices[model_name]
    model = model_choice((input_size, input_size), nb_classes)
    gpus = get_number_of_gpus()
    print('Fund {} gpus'.format(gpus))
    if gpus > 1:
        model = ModelMGPU(model, gpus)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=binary_soft_jaccard_loss,
        metrics=['acc', binary_jaccard_distance_rounded])

    model.load_weights(args.weights_path)
    generator = get_generator()
    # load the image
    image = np.array(Image.open(image_path))
    pred = predict_img_with_smooth_windowing(
        image,
        window_size=input_size,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=nb_classes,
        pred_func=(
            lambda img_batch_subdiv: model.predict(
                image_to_neural_input(img_batch_subdiv, generator), verbose=True
            )
        )
    )
    pred = np.argmax(pred, axis=2)
    pred_color = np.zeros((image.shape[0], image.shape[1], 3))

    if mask_path:
        mask = cv2.imread(mask_path, 0)
        mask = np.reshape(mask, (1, mask.shape[0], mask.shape[1]))
        p = np.reshape(pred, (1, pred.shape[0], pred.shape[1]))
        print('mIOU:', batch_classwise_general_jaccard(mask, p))
        print('F1:', f1_score(mask, p))

    class_color_map = {
        0: [237, 237, 237],  # Empty
        1: [254, 241, 179],  # Roads
        2: [116, 173, 209],  # Water
        3: [193, 235, 176],  # Grass
        4: [170, 170, 170]  # Buildings
    }

    for c in range(nb_classes):
        pred_color[:, :, 0] += ((pred[:, :] == c) * (class_color_map[c][2])).astype('uint8')
        pred_color[:, :, 1] += ((pred[:, :] == c) * (class_color_map[c][1])).astype('uint8')
        pred_color[:, :, 2] += ((pred[:, :] == c) * (class_color_map[c][0])).astype('uint8')

    out_path = os.path.join(output_path, file_name)
    print(cv2.imwrite(out_path, pred_color))

    cheap = cheap_tiling_prediction(image, window_size=input_size, nb_classes=nb_classes, pred_func=(
        lambda img_batch_subdiv: model.predict(
            image_to_neural_input(np.array(img_batch_subdiv), generator), verbose=True
        )
    ))

    cheap = np.argmax(cheap, axis=2)
    cheap_color = np.zeros((image.shape[0], image.shape[1], 3))

    if mask_path:
        mask = cv2.imread(mask_path, 0)
        mask = np.reshape(mask, (1, mask.shape[0], mask.shape[1]))
        p = np.reshape(cheap, (1, cheap.shape[0], cheap.shape[1]))
        print('mIOU:', batch_classwise_general_jaccard(mask, p))
        print('F1:', f1_score(mask, p))

    for c in range(nb_classes):
        cheap_color[:, :, 0] += ((cheap[:, :] == c) * (class_color_map[c][2])).astype('uint8')
        cheap_color[:, :, 1] += ((cheap[:, :] == c) * (class_color_map[c][1])).astype('uint8')
        cheap_color[:, :, 2] += ((cheap[:, :] == c) * (class_color_map[c][0])).astype('uint8')

    out_path = os.path.join(output_path, 'cheap-{}'.format(file_name))
    print(cv2.imwrite(out_path, cheap_color))


if __name__ == '__main__':
    run()
