import os
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping


def callbacks(logdir, weightsdir):
    weightsdir = os.path.join(weightsdir, 'weights.{epoch:02d}-{loss:.2f}.h5')

    checkpoint = ModelCheckpoint(weightsdir, monitor='val_acc', verbose=2,
                                 save_best_only=True, save_weights_only=False, mode='max')
    tensorboard_callback = TensorBoard(log_dir=logdir, write_graph=True, histogram_freq=0)
    plateau_callback = ReduceLROnPlateau(monitor='loss', factor=0.99, verbose=1, patience=0, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
    return [checkpoint, plateau_callback, tensorboard_callback, early_stopping]