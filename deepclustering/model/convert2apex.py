import contextlib
import warnings

import torch
from torch import Tensor

from .models import Model

try:
    from apex import amp
except ImportError:
    warnings.warn("Apex not installed, using PyTorch default setting.", RuntimeWarning)
    amp = None


def to_Apex(model: Model, opt_level=None, verbosity=0, **kwargs) -> Model:
    # consider the apex model
    if opt_level is None:
        # no action is taken.
        return model
    try:
        # try to convert to apex model.
        orig_device: torch.device = model.torchnet.parameters().__next__().device
        model.to(torch.device("cuda"))
        model.torchnet, model.optimizer = amp.initialize(
            model.torchnet,
            model.optimizer,
            opt_level=opt_level,
            verbosity=verbosity,
            **kwargs,
        )
        model.to(orig_device)
        model.is_apex = True
    except Exception as e:
        # nothing happens.
        warnings.warn(f"`to_Apex` fails with error message: {e}", RuntimeWarning)
        assert model.is_apex is False
    finally:
        return model


@contextlib.contextmanager
def AMPGradientBackwardStep(loss: Tensor, model: Model):
    """
    Being called when a Model is wrapped by apex model
    1. initialize: model.zero_grad
    2. return amp.scaled_loss as loss
    3. optimizer.step
    :param loss:
    :param model:
    :return:
    """
    model.zero_grad()
    with amp.scale_loss(loss, model.optimizer) as scaled_loss:
        yield scaled_loss
    model.step()
