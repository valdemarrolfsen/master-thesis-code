import numpy as np
from keras import backend as K


def boolean_jaccard(y_true, y_pred):
    smooth = K.epsilon()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

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
