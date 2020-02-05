from .models import Model


class EMA_Model:
    def __init__(self, model: Model, alpha=0.999) -> None:
        super().__init__()
        # here we deepcopy a `Model`, including the torchmodel, optimizer and, scheduler
        self.model = Model.initialize_from_state_dict(model.state_dict())
        self.alpha = alpha
        self.global_step = 0
        # detach the param for the ema model
        for param in self.model.torchnet.parameters():
            param.detach_()
        self.train()

    def step(self, student_model: Model):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for ema_param, s_param in zip(
            self.model.torchnet.parameters(), student_model._torchnet.parameters()
        ):
            ema_param.data.mul_(alpha).add_(1 - alpha, s_param.data)
        self.global_step += 1

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def set_mode(self, mode):
        self.model.set_mode(mode)

    # this would enable the framework to automatically load the state_dict
    def state_dict(self):
        """
        enable save and load
        :return:
        """

        return {
            **self.model.state_dict(),
            **{"alpha": self.alpha, "global_step": self.global_step},
        }

    def load_state_dict(self, state_dict: dict):
        self.alpha = state_dict["alpha"]
        self.global_step = state_dict["global_step"]
        del state_dict["alpha"]
        del state_dict["global_step"]
        self.model.load_state_dict(state_dict)

    @property
    def training(self):
        return self.model.training

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        return self.model.to(device)
