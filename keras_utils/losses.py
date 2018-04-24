from keras import backend as K


def jaccard_distance(y_true, y_pred):
    smooth = K.epsilon()
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1) - intersection
    jac = (intersection + smooth) / (union + smooth)
    return jac


def binary_jaccard_distance(y_true, y_pred):
    smooth = K.epsilon()
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(K.abs(y_true) + K.abs(y_pred)) - intersection
    jac = (intersection + smooth) / (union + smooth)
    return jac


def soft_jaccard_loss(y_true, y_pred):
    return -K.log(jaccard_distance(y_true, y_pred)) + K.categorical_crossentropy(y_true, y_pred)


def binary_jaccard_loss(target, output):
    return 1 - binary_jaccard_distance(target, output)


def binary_soft_jaccard_loss(target, output):
    return -K.log(jaccard_distance(target, output)) + K.binary_crossentropy(target, output)
