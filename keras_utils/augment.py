import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)
np.random.seed(1)

augmentations = [
    {'common': True, 'seq': iaa.Flipud(1)},

    {'common': True, 'seq': iaa.Fliplr(1)},

    {'common': False, 'seq': iaa.GaussianBlur(sigma=(0.0, 3.0))},

    {'common': False, 'seq': iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))}
]


def image_augmentation(img, mask):
    for aug in augmentations:
        if not aug['common']:
            img = aug['seq'].augment_image(img)
        elif np.random.rand() > 0.5:
            img = aug['seq'].augment_image(img)
            mask = aug['seq'].augment_image(mask)

    return img, mask
