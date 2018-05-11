import numpy as np
from keras import backend as K

from keras_utils.generators import create_generator
from keras_utils.metrics import batch_general_jaccard, f1_score, maximize_threshold
from networks.unet.unet import build_unet

from run_unet_experiments import experiments


def run():
    images_path = '/data/buildings/test'
    input_size = (320, 320)
    batch_size = 2600
    weights = 'weights_train/weights.unet-experiment-{}.h5'

    generator, _ = create_generator(
        images_path,
        input_size,
        batch_size,
        1,
        rescale=False,
        binary=True
    )

    images, masks = next(generator)

    for i, options in enumerate(experiments):
        print('Running prediction for experiment {}'.format(str(i)))
        dropout = options['dropout']

        model = build_unet(input_size, nb_classes=1, dropout=dropout)
        model.compile('sgd', 'mse')
        model.load_weights(weights.format(i))

        probs = model.predict(images, verbose=1)

        iou = batch_general_jaccard(masks, probs, binary=True)
        f1 = K.eval(f1_score(K.variable(masks), K.variable(probs)))
        print('mean IOU for {}: {}'.format(i, np.mean(iou)))
        print('F1 score for {}: {}'.format(i, f1))

        K.clear_session()


if __name__ == '__main__':
    run()
