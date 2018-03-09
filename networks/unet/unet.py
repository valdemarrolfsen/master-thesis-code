from keras import layers, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, concatenate, Flatten, Dense, \
    BatchNormalization, Activation, Dropout
from keras.optimizers import Adam


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


def build_unet(nb_classes, input_shape):

    concat_axis = 3
    inputs = layers.Input((input_shape[0], input_shape[1], 3))

    conv1 = Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="relu", data_format="channels_last", kernel_initializer='he_uniform')(inputs)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(pool1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(pool2)
    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(pool3)
    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(pool4)
    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv5)

    convbott = Conv2D(1024, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(pool5)
    convbott = Conv2D(1024, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(convbott)
    convbott = Dropout(0.5)(convbott)
    convbott = BatchNormalization(axis=-1)(convbott)
    up_convbott = UpSampling2D(size=(2, 2), data_format="channels_last")(convbott)

    ch, cw = get_crop_shape(conv5, up_convbott)
    crop_conv5 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv5)
    upbott = concatenate([up_convbott, crop_conv5], axis=concat_axis)
    convbott1 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(upbott)
    convbott1 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(convbott1)
    up_convbott1 = UpSampling2D(size=(2, 2), data_format="channels_last")(convbott1)

    ch, cw = get_crop_shape(conv4, up_convbott1)
    crop_conv4 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv4)
    up6 = concatenate([up_convbott1, crop_conv4], axis=concat_axis)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(up6)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(conv6)
    up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv6)

    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(up7)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(conv7)
    up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv7)

    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(up8)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(conv8)
    up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv8)

    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(up9)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last",  kernel_initializer='he_uniform')(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = layers.Conv2D(nb_classes, (1, 1))(conv9)
    act = Activation('softmax')(conv10)
    model = Model(inputs=inputs, outputs=act)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
