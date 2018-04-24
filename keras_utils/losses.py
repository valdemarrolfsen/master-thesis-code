from keras import backend as K


def jaccard_distance(target, output):
    smooth = K.epsilon()
    intersection = K.sum(K.abs(target * output), axis=-1)
    union = K.sum(K.abs(target) + K.abs(output), axis=-1) - intersection
    jac = (intersection + smooth) / (union + smooth)
    return jac


def binary_jaccard_distance(target, output):
    smooth = K.epsilon()
    intersection = K.sum(target * output, axis=[0, -1, -2])
    sum_ = K.sum(target + output, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def soft_jaccard_loss(target, output):
    return -K.log(jaccard_distance(target, output)) + K.categorical_crossentropy(target, output)


def binary_jaccard_loss(target, output):
    return 1 - binary_jaccard_distance(target, output)


def binary_soft_jaccard_loss(target, output):
    return -K.log(binary_jaccard_distance(target, output)) + K.binary_crossentropy(target, output)
