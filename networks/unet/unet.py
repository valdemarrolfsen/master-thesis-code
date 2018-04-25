from keras import backend as K
from keras import layers, Model
from keras.applications import VGG16
from keras.layers import Conv2D, MaxPooling2D, concatenate, BatchNormalization, Activation, \
    Dropout, Conv2DTranspose
from keras.optimizers import Adam

from keras_utils.losses import soft_jaccard_loss, binary_soft_jaccard_loss
from keras_utils.multigpu import ModelMGPU


def down_block(input_tensor, filters, bottleneck=False):
    name = 'Bottleneck' if bottleneck else 'DownBlock'
    with K.name_scope(name):
        x = Conv2D(filters, (3, 3), padding="same", activation="relu", data_format="channels_last", kernel_initializer='he_uniform')(input_tensor)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(filters, (3, 3), padding="same", activation="relu", data_format="channels_last", kernel_initializer='he_uniform')(x)
        x = BatchNormalization(axis=-1)(x)
    return x


def up_block(input_tensor, concat_target, filters):
    concat_axis = 3
    with K.name_scope('UpBlock'):
        x = concatenate([Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(input_tensor), concat_target], axis=concat_axis)
        x = Dropout(0.2)(x)
        x = Conv2D(filters, (3, 3), padding="same", activation="relu", data_format="channels_last", kernel_initializer='he_uniform')(x)
        x = Conv2D(filters, (3, 3), padding="same", activation="relu", data_format="channels_last", kernel_initializer='he_uniform')(x)
    return x


def build_unet(input_shape, nb_classes, lr=1e-4):
    inputs = layers.Input((input_shape[0], input_shape[1], 3))
    conv1 = down_block(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)

    conv2 = down_block(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

    conv3 = down_block(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

    conv4 = down_block(pool3, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

    conv5 = down_block(pool4, 512)
    pool5 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv5)

    bottleneck = down_block(pool5, 1024, bottleneck=True)

    conv6 = up_block(bottleneck, conv5, 512)
    conv7 = up_block(conv6, conv4, 256)
    conv8 = up_block(conv7, conv3, 128)
    conv9 = up_block(conv8, conv2, 64)
    conv10 = up_block(conv9, conv1, 32)
    conv11 = layers.Conv2D(nb_classes, (1, 1))(conv10)

    if nb_classes == 1:
        activation = 'sigmoid'
        loss = binary_soft_jaccard_loss
    else:
        activation = 'softmax'
        loss = soft_jaccard_loss

    act = Activation(activation)(conv11)
    model = Model(inputs=inputs, outputs=act)
    model = ModelMGPU(model, 2)
    model.compile(
        optimizer=Adam(lr=lr),
        loss=loss,
        metrics=['acc'])

    return model


def build_unet16(input_shape, nb_classes, lr=1e-4):
    """
    Build a Unet16 with a VGG16 encoder pretrained on the imagenet dataset
    """
    base_model = VGG16(weights='imagenet', include_top=False)
    inputs = layers.Input((input_shape[0], input_shape[1], 3))

    # Block 1
    x = base_model.get_layer('block1_conv1')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = base_model.get_layer('block1_conv2')(x)
    x1 = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x1)

    # Block 2
    x = base_model.get_layer('block2_conv1')(x)
    x = BatchNormalization(axis=-1)(x)
    x = base_model.get_layer('block2_conv2')(x)
    x2 = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x2)

    # Block 3
    x = base_model.get_layer('block3_conv1')(x)
    x = BatchNormalization(axis=-1)(x)
    x = base_model.get_layer('block3_conv2')(x)
    x = BatchNormalization(axis=-1)(x)
    x = base_model.get_layer('block3_conv3')(x)
    x3 = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x3)

    # Block 4
    x = base_model.get_layer('block4_conv1')(x)
    x = BatchNormalization(axis=-1)(x)
    x = base_model.get_layer('block4_conv2')(x)
    x = BatchNormalization(axis=-1)(x)
    x = base_model.get_layer('block4_conv3')(x)
    x4 = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x4)

    # Block 5
    x = base_model.get_layer('block5_conv1')(x)
    x = BatchNormalization(axis=-1)(x)
    x = base_model.get_layer('block5_conv2')(x)
    x = BatchNormalization(axis=-1)(x)
    x = base_model.get_layer('block5_conv3')(x)
    x5 = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x5)

    # Bottleneck
    x = down_block(x, 1024, bottleneck=True)

    x = up_block(x, x5, 512)

    x = up_block(x, x4, 512)

    x = up_block(x, x3, 256)

    x = up_block(x, x2, 128)

    x = up_block(x, x1, 64)

    x = layers.Conv2D(nb_classes, (1, 1))(x)

    if nb_classes == 1:
        activation = 'sigmoid'
        loss = binary_soft_jaccard_loss
    else:
        activation = 'softmax'
        loss = soft_jaccard_loss

    act = Activation(activation)(x)
    model = Model(inputs=inputs, outputs=act)
    model = ModelMGPU(model, 2)
    model.compile(
        optimizer=Adam(lr=lr),
        loss=loss,
        metrics=['acc'])

    return model
