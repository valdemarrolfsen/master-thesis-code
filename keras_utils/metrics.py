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


def classwise_general_jaccard(y_true, y_pred):
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
    return result


def batch_general_jaccard(y_true, y_pred):
    batch_result = []
    for true, pred in zip(y_true, y_pred):
        batch_result.append(general_jaccard(true, pred))
    return batch_result


def batch_classwise_general_jaccard(y_true, y_pred):
    batch_result = []
    for true, pred in zip(y_true, y_pred):
        batch_result.append(classwise_general_jaccard(true, pred))
    return np.mean(batch_result, axis=0)


def f1_score(y_true, y_pred):
    """https://stackoverflow.com/questions/45411902/how-to-use-f1-score-with-keras-model"""
    c1 = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    c2 = np.sum(np.round(np.clip(y_pred, 0, 1)))
    c3 = np.sum(np.round(np.clip(y_true, 0, 1)))
    if c3 == 0:
        return 0
    precision = c1 / c2
    recall = c1 / c3
    return 2 * (precision * recall) / (precision + recall)


def boolean_f1(y_true, y_pred):
    smooth = K.epsilon()
    c1 = (y_true * y_pred).sum()
    c2 = np.sum(y_pred)
    c3 = np.sum(y_true)
    if c3 == 0:
        return 0
    precision = c1 + smooth / c2 + smooth
    recall = c1 / c3
    return 2 * (precision * recall) / (precision + recall)


def batch_classwise_f1_score(y_true, y_pred):
    batch_result = []
    for true, pred in zip(y_true, y_pred):
        batch_result.append(classwise_f1_score(true, pred))
    return np.mean(batch_result, axis=0)


def classwise_f1_score(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for cls in set(y_true.flatten()):
        if cls == 0:
            continue
        result += [boolean_f1(y_true == cls, y_pred == cls)]
    return result


def binary_jaccard_distance_rounded(target, output):
    smooth = K.epsilon()
    output = K.round(K.clip(output, 0, 1))
    intersection = K.sum(target * output, axis=[0, -1, -2])
    sum_ = K.sum(target + output, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)
