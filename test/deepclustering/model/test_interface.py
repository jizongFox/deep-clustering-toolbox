from unittest import TestCase

import torch
from deepclustering.model.models import Model


class TestModel_(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._img = torch.randn(1, 3, 224, 224)

    def test_initialization_from_config(self):
        arch = {"name": "enet", "num_classes": 10, "input_dim": 3}
        optim = {"name": "Adam", "lr": 0.1}
        scheduler = {"name": "StepLR", "step_size": 10}
        model = Model(arch=arch, optimizer=optim, scheduler=scheduler)
        model.initialize_from_state_dict(model.state_dict())
        print(model(self._img).shape)

    def test_initializaton_from_modules(self):
        import torchvision.models as models

        arch = models.resnet18(pretrained=True)
        optim = torch.optim.Adam(arch.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10)
        model = Model(arch, optim, scheduler)
        print(model(self._img).shape)

    def test_initialization_from_mix(self):
        import torchvision.models as models

        arch = models.resnet18(pretrained=True)
        optim = {"name": "Adam", "lr": 0.1}
        scheduler = {"name": "StepLR", "step_size": 10}
        model = Model(arch, optim, scheduler)
        print(model(self._img).shape)

    def test_initialization_lack_of_some_parts(self):
        arch = {"name": "enet", "num_classes": 10, "input_dim": 3}
        optim = {"name": "Adam", "lr": 0.1}
        model = Model(arch=arch, optimizer=optim)
        model.initialize_from_state_dict(model.state_dict())
        print(model(self._img).shape)

    def test_load_state_dict(self):
        arch = {"name": "enet", "num_classes": 10, "input_dim": 3}
        optim = {"name": "Adam", "lr": 0.1}
        model1 = Model(arch=arch, optimizer=optim)
        model2 = Model(arch=arch, optimizer=optim)
        model2.load_state_dict(model1.state_dict())
