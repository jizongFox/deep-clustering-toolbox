import numpy as np


class WeightScheduler(object):
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
        """Returns the state of the weight_scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): weight_scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    @staticmethod
    def get_lr(**kwargs):
        raise NotImplementedError

    def plot_weights(self):
        _current_epoch = self.epoch
        self.epoch = 0
        epochs = list(range(int(self.max_epoch * 1.5)))
        lrs = []
        for _ in epochs:
            lrs.append(self.value)
            self.step()
        assert len(lrs) == len(epochs)
        import matplotlib  # type: ignore

        _current_bkend = matplotlib.get_backend()
        try:
            matplotlib.use("tkagg")
        except:
            pass
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(f"{self.__class__.__name__}, learning (weight) rate")
        plt.show()
        plt.pause(2)

        # return back
        self.epoch = _current_epoch
        try:
            matplotlib.use(_current_bkend)
        except:
            pass


class RampScheduler(WeightScheduler):
    def __init__(
        self, begin_epoch=0, max_epoch=10, min_value=0.0, max_value=1.0, ramp_mult=-5.0
    ):
        super().__init__()
        self.begin_epoch = int(begin_epoch)
        self.max_epoch = int(max_epoch)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.mult = float(ramp_mult)
        self.epoch = 0

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.get_lr(
            self.epoch,
            self.begin_epoch,
            self.max_epoch,
            self.min_value,
            self.max_value,
            self.mult,
        )

    @staticmethod
    def get_lr(epoch, begin_epoch, max_epochs, min_val, max_val, mult):
        if epoch < begin_epoch:
            return min_val
        elif epoch >= max_epochs:
            return max_val
        return min_val + (max_val - min_val) * np.exp(
            mult * (1.0 - float(epoch - begin_epoch) / (max_epochs - begin_epoch)) ** 2
        )


class ConstantScheduler(WeightScheduler):
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
