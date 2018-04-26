import numpy as np
import os
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping


def callbacks(logdir, weightsdir, filename, monitor_val='val_acc'):
    weightsdir = os.path.join(weightsdir, 'weights.{}.h5'.format(filename))

    checkpoint = ModelCheckpoint(weightsdir, monitor=monitor_val, verbose=2,
                                 save_best_only=True, save_weights_only=True, mode='auto')
    tensorboard_callback = TensorBoard(log_dir=logdir, write_graph=True, histogram_freq=0)
    plateau_callback = ReduceLROnPlateau(monitor=monitor_val, factor=np.sqrt(0.1), verbose=1, patience=3, min_lr=0.5e-6)
    early_stopping = EarlyStopping(monitor=monitor_val, patience=50, verbose=1)
    return [checkpoint, plateau_callback, tensorboard_callback, early_stopping]
