from enum import Enum


class ModelMode(Enum):
    """ Different mode of model """
    TRAIN = 'TRAIN'  # during training
    EVAL = 'EVAL'  # eval mode. On validation data
    PRED = 'PRED'

    @staticmethod
    def from_str(mode_str):
        """ Init from string
            :param mode_str: ['train', 'eval', 'predict']
        """
        if mode_str == 'train':
            return ModelMode.TRAIN
        elif mode_str == 'eval':
            return ModelMode.EVAL
        elif mode_str == 'predict':
            return ModelMode.PRED
        else:
            raise ValueError('Invalid argument mode_str {}'.format(mode_str))
