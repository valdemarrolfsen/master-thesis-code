from keras import backend as K

from keras_utils.metrics import jaccard_distance, jaccard_distance_loss


def _end_mean(x, axis=-1):
    """ Same as K.mean, but defaults to the final axis.
    """
    return K.mean(x, axis=axis)


def _metric_2d_adaptor(y_true, y_pred, metric=None, summary=_end_mean, **kwargs):
    """ Adapt a one dimensional loss function to work with 2d segmentation data.
    """
    if metric is None:
        raise ValueError("You must provide a metric function such as binary_crossentropy")
    pred_shape = K.shape(y_pred)
    true_shape = K.shape(y_true)
    y_pred_reshaped = K.reshape(y_pred, (-1, pred_shape[-1]))
    y_true_reshaped = K.reshape(y_true, (-1, true_shape[-1]))

    result = metric(y_pred_reshaped, y_true_reshaped, **kwargs)

    if summary is not None:
        result = summary(result)

    if len(true_shape) >= 3:
        return K.reshape(result, true_shape[:-1])
    else:
        return result


def mean_intersection_over_union(y_true, y_pred):
    """ Same as keras_contrib.losses.jaccard_distance for 2d label data with one-hot channels.
    """
    return _metric_2d_adaptor(y_true, y_pred, metric=jaccard_distance_loss, summary=_end_mean)
