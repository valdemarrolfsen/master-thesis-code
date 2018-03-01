import keras.backend as K
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


def callbacks(logdir):
    model_checkpoint = ModelCheckpoint("weights_train/weights.{epoch:02d}-{loss:.2f}.h5", monitor='loss', verbose=1,
                                       period=1)
    tensorboard_callback = TensorBoard(log_dir=logdir, write_graph=True, write_images=True, histogram_freq=1)
    plateau_callback = ReduceLROnPlateau(monitor='loss', factor=0.99, verbose=1, patience=0, min_lr=0.00001)
    return [model_checkpoint, plateau_callback, tensorboard_callback]
