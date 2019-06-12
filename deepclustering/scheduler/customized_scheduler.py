import torch
import numpy as np


class Scheduler(object):
    def __init__(self):
        pass

    def get_current_value(self):
        return NotImplementedError

    def step(self):
        return NotImplementedError

    @property
    def value(self):
        # return self.value
        return NotImplementedError

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    @staticmethod
    def get_lr(**kwargs):
        raise NotImplementedError


class RampScheduler(Scheduler):

    def __init__(self, begin_epoch, max_epoch, max_value, ramp_mult):
        super().__init__()
        self.begin_epoch = int(begin_epoch)
        self.max_epoch = int(max_epoch)
        self.max_value = float(max_value)
        self.mult = float(ramp_mult)
        self.epoch = 0

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.get_lr(self.epoch, self.begin_epoch, self.max_epoch, self.max_value, self.mult)

    @staticmethod
    def get_lr(epoch, begin_epoch, max_epochs, max_val, mult):
        if epoch < begin_epoch:
            return 0.
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1. - float(epoch - begin_epoch) / (max_epochs - begin_epoch)) ** 2)


class ConstantScheduler(Scheduler):

    def __init__(self, begin_epoch, max_value=1.0):
        super().__init__()
        self.begin_epoch = int(begin_epoch)
        self.max_value = float(max_value)
        self.epoch = 0

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.get_lr(self.epoch, self.begin_epoch, self.max_value)

    @staticmethod
    def get_lr(epoch, begin_epoch, max_value):
        if epoch < begin_epoch:
            return 0.0
        else:
            return max_value


class RampDownScheduler(Scheduler):

    def __init__(self, max_epoch, max_value, ramp_mult, min_val, cutoff):
        super().__init__()
        self.max_epoch = int(max_epoch)
        self.max_value = float(max_value)
        self.mult = float(ramp_mult)
        self.epoch = 0
        self.min_val = float(min_val)
        self.cutoff = int(cutoff)

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.ramp_down(self.epoch, self.max_epoch, self.max_value, self.mult, self.min_val, self.cutoff)

    @staticmethod
    def ramp_down(epoch, max_epochs, max_val, mult, min_val, cutoff):
        assert cutoff < max_epochs
        if epoch == 0:
            return max_val
        elif epoch >= cutoff:
            return min_val
        return max_val - max_val * np.exp(mult * (1. - float(epoch) / (cutoff)) ** 2) +min_val
