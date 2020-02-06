from torch import Tensor

from .convert2apex import AMPGradientBackwardStep, to_Apex
from .models import Model, NormalGradientBackwardStep
from .ema import EMA_Model


def ZeroGradientBackwardStep(loss: Tensor, model: Model):
    """
    This context manager takes loss and model as the input.
    if the model is wrapped by the AMP package, the official AMP
    context manager is called.
    if the model is traditional model, a context manager doing
    zero_grad and step would perform at the init and exit.
    In both case, No longer need to type zero_grad and step.

    >>> with ZeroGradientBackwardStep(loss, model) as scaled_loss:
    >>>     scaled_loss.backward()
    :param loss: Tensor loss to call backward()
    :param model: self-defined Model wrapper
    """
    if hasattr(model, "is_apex"):
        if model.is_apex:
            return AMPGradientBackwardStep(loss, model)
    return NormalGradientBackwardStep(loss, model)
