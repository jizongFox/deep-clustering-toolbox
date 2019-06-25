import warnings

import torch
from torch import Tensor
from apex import amp
from .general import Model


def to_Apex(model: Model, opt_level=None, **kwargs) -> Model:
    # consider the apex model
    if opt_level is None:
        # no action is taken.
        return model
    try:
        # try to convert to apex model.
        orig_device: torch.device = model.torchnet.parameters().__next__().device
        model.to(torch.device('cuda'))
        model.torchnet, model.optimizer = amp.initialize(
            model.torchnet, model.optimizer,
            opt_level=opt_level,
            # loss_scale="1280.0",
            **kwargs
        )
        model.to(orig_device)
        model.is_apex = True
    except Exception as e:
        # nothing happens.
        warnings.warn(f'to_apex fails with eror message: {e}', RuntimeWarning)
        assert model.is_apex is False
    finally:
        return model


def AMPGradientBackwardStep(loss: Tensor, model: Model):
    model.zero_grad()
    return amp.scale_loss(loss, model.optimizer)
