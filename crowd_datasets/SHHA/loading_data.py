import torchvision.transforms as standard_transforms
from .SHHA import SHHA
from .crowd import Crowd
import os


# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def loading_data(data_root):
    # create the training dataset
    train_set = Crowd(os.path.join(data_root, 'train'),
                                  256,
                                  8,
                                  'train')
    val_set = Crowd(os.path.join(data_root, 'val'),
                                  256,
                                  8,
                                  'val')

    return train_set, val_set
