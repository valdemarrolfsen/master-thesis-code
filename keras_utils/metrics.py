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


def batch_general_jaccard(y_true, y_pred, binary=False, threshold=0.5):
    batch_result = []

    print(threshold)

    for i in range(len(y_true)):
        if binary:
            pred = np.zeros(y_pred[i].shape)
            pred[:][:] = y_pred[i][:][:] > threshold
            true = y_true[i]
        else:
            pred = np.argmax(y_pred[i], axis=2)
            true = np.argmax(y_true[i], axis=2)
        batch_result.append(general_jaccard(true, pred))
    return batch_result


def f1_score(y_true, y_pred):
    """https://stackoverflow.com/questions/45411902/how-to-use-f1-score-with-keras-model"""
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    if c3 == 0:
        return 0
    precision = c1 / c2
    recall = c1 / c3
    return 2 * (precision * recall) / (precision + recall)


def maximize_threshold(y_true, y_pred):
    threshold_step = 0.1
    thresholds = np.arange(0.0, 1.0, threshold_step)

    print(thresholds)

    mean_IOUs = [np.mean(batch_general_jaccard(y_true, y_pred, True, step)) for step in thresholds]
    max_IOU_index = np.argmax(mean_IOUs)

    print("Max IOU with threshold {}".format(threshold_step*max_IOU_index))
    print("Max IOU is {}".format(mean_IOUs[max_IOU_index]))
    print(mean_IOUs)
    return threshold_step*max_IOU_index


def binary_jaccard_distance_rounded(target, output):
    smooth = K.epsilon()
    output = K.round(K.clip(output, 0, 1))
    intersection = K.sum(target * output, axis=[0, -1, -2])
    sum_ = K.sum(target + output, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)
