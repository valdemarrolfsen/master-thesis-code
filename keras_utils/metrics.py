import numpy as np
from keras import backend as K
import tensorflow as tf


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


def jaccard_without_background(target, output):
    """

    Args:
        output(tensor): Tensor of shape (batch_size,w,h,num_classes). Output of SOFTMAX Activation
        target: Tensor of shape (batch_size,w,h,num_classes). one hot encoded class matrix

    Returns:
        jaccard estimation
    """
    smooth = K.epsilon()
    output = output[:, :, :, 1:]
    target = target[:, :, :, 1:]
    output = K.clip(K.abs(output), K.epsilon(), 1. - K.epsilon())
    target = K.clip(K.abs(target), K.epsilon(), 1. - K.epsilon())

    union = K.sum(output + target, axis=-1)
    intersection = K.sum(output * target, axis=-1)

    iou = (intersection + smooth) / (union - intersection + smooth)
    return iou


def mean_intersection_over_union(y_true, y_pred, smooth=None, axis=-1):
    """Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Also see jaccard which takes a slighty different approach.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    if smooth is None:
        smooth = K.epsilon()
    pred_shape = K.shape(y_pred)
    true_shape = K.shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_pred_reshaped = K.reshape(y_pred, (-1, pred_shape[-1]))
    y_true_reshaped = K.reshape(y_true, (-1, true_shape[-1]))

    # correctly classified
    clf_pred = K.one_hot(K.argmax(y_pred_reshaped), num_classes=true_shape[-1])
    equal_entries = K.cast(
        K.equal(clf_pred, y_true_reshaped), dtype='float32') * y_true_reshaped

    intersection = K.sum(equal_entries, axis=1)
    union_per_class = K.sum(
        y_true_reshaped, axis=1) + K.sum(
            y_pred_reshaped, axis=1)

    # smooth added to avoid dividing by zero
    iou = (intersection + smooth) / (
        (union_per_class - intersection) + smooth)

    return K.mean(iou)
