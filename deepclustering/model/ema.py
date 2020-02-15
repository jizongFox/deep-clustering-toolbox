from .models import Model


class EMA_Model:
    def __init__(self, model: Model, alpha=0.999, weight_decay=0.0) -> None:
        super().__init__()
        # here we deepcopy a `Model`, including the torchmodel, optimizer and, scheduler
        # self._model = Model.initialize_from_state_dict(model.state_dict())
        self._model = model
        self._alpha = alpha
        self._weight_decay = weight_decay
        self._global_step = 0
        # detach the param for the ema model
        for param in self._model._torchnet.parameters():
            param.detach_()
        self.train()

    def step(self, student_model: Model):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self._global_step + 1), self._alpha)
        for ema_param, s_param in zip(
            self._model._torchnet.parameters(), student_model._torchnet.parameters()
        ):
            ema_param.data.mul_(alpha).add_(1 - alpha, s_param.data)
            if self._weight_decay > 0:
                ema_param.data.mul_(1 - self._weight_decay)
        # running mean and vars for bn
        for (name, ema_buffer), (_, s_buffer) in zip(
            self._model._torchnet.named_buffers(),
            student_model._torchnet.named_buffers(),
        ):
            if "running_mean" in name or "running_var" in name:
                ema_buffer.data.mul_(alpha).add_(1 - alpha, s_buffer.data)
                if self._weight_decay > 0:
                    ema_buffer.data.mul_(1 - self._weight_decay)
        self._global_step += 1

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def set_mode(self, mode):
        self._model.set_mode(mode)

    # this would enable the framework to automatically load the state_dict
    def state_dict(self):
        """
        enable save and load
        :return:
        """

        return {
            **self._model.state_dict(),
            **{"alpha": self._alpha, "global_step": self._global_step},
        }

    def load_state_dict(self, state_dict: dict):
        self._alpha = state_dict["alpha"]
        self._global_step = state_dict["global_step"]
        del state_dict["alpha"]
        del state_dict["global_step"]
        self._model.load_state_dict(state_dict)

    @property
    def training(self):
        return self._model.training

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def to(self, device):
        return self._model.to(device)
