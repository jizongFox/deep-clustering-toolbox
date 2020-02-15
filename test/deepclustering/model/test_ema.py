from unittest import TestCase

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms import ToTensor

from deepclustering.model import Model, EMA_Model


class TestEMA(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._model = Model(arch=deeplabv3_resnet101(pretrained=True))
        self._model_ema = EMA_Model(
            Model(deeplabv3_resnet101(False)), alpha=0.9, weight_decay=1e-4
        )
        # self._model_ema._model.load_state_dict(self._model.state_dict())
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._img = (
            ToTensor()(Image.open("img1.jpg").convert("RGB"))
            .unsqueeze(0)
            .to(self._device)
        )

        self._model.to(self._device)
        self._model_ema.to(self._device)

    def test_1(self):
        self._model.eval()
        self._model_ema.eval()
        with torch.no_grad():
            student_prediction = self._model(self._img)["out"]
            plt.figure(1)
            plt.imshow(self._img[0].cpu().numpy().transpose(1, 2, 0))
            plt.contour(student_prediction.max(1)[1].cpu().detach().numpy()[0])
            plt.show(block=False)
            self._model_ema._global_step += 1
            for i in range(1000):
                teacher_prediction = self._model_ema(self._img)["out"]
                plt.figure(2)
                plt.clf()
                plt.imshow(self._img[0].cpu().numpy().transpose(1, 2, 0))
                plt.contour(teacher_prediction.max(1)[1].cpu().detach().numpy()[0])
                plt.show(block=False)
                plt.pause(0.00000003)
                self._model_ema.step(self._model)
                # student_state_dict = self._model._torchnet.state_dict()
                # teacher_state_dict = self._model_ema._model._torchnet.state_dict()
                # for k in student_state_dict.keys():
                #     assert torch.allclose(student_state_dict[k], teacher_state_dict[k]), k
