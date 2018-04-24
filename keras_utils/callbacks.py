import numpy as np
import os
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping, Callback

from keras_utils.metrics import batch_general_jaccard


class MeanIOUCallback(Callback):
    def __init__(self):
        super().__init__()
        self.ious = []

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        generator = self.validation_data
        images, masks = next(generator)
        probs = self.model.predict(images, verbose=0)
        iou = np.mean(batch_general_jaccard(masks, probs, binary=True))
        self.ious.append(iou)
        print('- mean IOU: {}'.format(iou))
        return


def callbacks(logdir, weightsdir, filename, monitor_val='val_acc'):
    weightsdir = os.path.join(weightsdir, 'weights.{}.h5'.format(filename))

    checkpoint = ModelCheckpoint(weightsdir, monitor=monitor_val, verbose=2,
                                 save_best_only=True, save_weights_only=True, mode='auto')
    tensorboard_callback = TensorBoard(log_dir=logdir, write_graph=True, histogram_freq=0)
    plateau_callback = ReduceLROnPlateau(monitor=monitor_val, factor=np.sqrt(0.1), verbose=1, patience=3, min_lr=0.5e-6)
    early_stopping = EarlyStopping(monitor=monitor_val, patience=10, verbose=1)
    meaniou = MeanIOUCallback()
    return [checkpoint, plateau_callback, tensorboard_callback, early_stopping, meaniou]
