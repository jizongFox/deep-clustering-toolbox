from .general import Model, NormalGradientBackwardStep
from .convert2apex import AMPGradientBackwardStep, to_Apex


def GradientBackwardStep(loss, model: Model):
    if hasattr(model, 'is_apex'):
        if model.is_apex:
            return AMPGradientBackwardStep(loss, model)
    return NormalGradientBackwardStep(loss, model)
