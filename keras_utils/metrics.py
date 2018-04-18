import numpy as np
from keras import backend as K


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
        result += [jaccard(y_true == cls, y_pred == cls)]

    return np.mean(result)


def batch_general_jaccard(y_true, y_pred):
    batch_result = []
    pred_shape = K.eval(K.shape(y_pred))
    for i in range(pred_shape[0]):
        pred = np.argmax(y_pred[i], axis=2)
        true = np.argmax(y_true[i], axis=2)
        batch_result.append(general_jaccard(true, pred))
    return batch_result


def jaccard(y_true, y_pred):
    smooth = K.epsilon()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard_distance_loss(y_true, y_pred):
    """Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight.
    For example, assume you are trying to predict if each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    smooth = K.epsilon()
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1) - intersection
    jac = (intersection + smooth) / (union + smooth)
    return (1 - jac) * smooth


def jaccard_distance(y_true, y_pred):
    smooth = K.epsilon()
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1) - intersection
    jac = (intersection + smooth) / (union + smooth)
    return jac


def soft_jaccard_loss(y_true, y_pred):
    return -K.log(jaccard_distance(y_true, y_pred)) + K.categorical_crossentropy(y_pred, y_true)


def mean_jaccard_loss(y_true, y_pred):
    mean_jac = np.mean(batch_general_jaccard(y_true, y_pred))
    return 1 - mean_jac
