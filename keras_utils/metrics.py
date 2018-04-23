import numpy as np
from keras import backend as K


def boolean_jaccard(y_true, y_pred):
    smooth = K.epsilon()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def general_jaccard(y_true, y_pred):
    result = []
    for cls in set(y_true.flatten()):
        if cls == 0:
            continue
        result += [boolean_jaccard(y_true == cls, y_pred == cls)]

    return np.mean(result)


def batch_general_jaccard(y_true, y_pred, binary=False):
    batch_result = []
    for i in range(len(y_true)):
        if binary:
            pred = np.round(y_pred[i])
            true = y_true[i]
        else:
            pred = np.argmax(y_pred[i], axis=2)
            true = np.argmax(y_true[i], axis=2)
        batch_result.append(general_jaccard(true, pred))
    return batch_result


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
