from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping


def callbacks(logdir):
    model_checkpoint = ModelCheckpoint("weights_train/weights.{epoch:02d}-{loss:.2f}.h5", monitor='loss', verbose=1,
                                       period=1)
    tensorboard_callback = TensorBoard(log_dir=logdir, write_graph=True, histogram_freq=0)
    plateau_callback = ReduceLROnPlateau(monitor='loss', factor=0.99, verbose=1, patience=0, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
    return [model_checkpoint, plateau_callback, tensorboard_callback, early_stopping]