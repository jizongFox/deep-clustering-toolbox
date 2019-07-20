from unittest import TestCase
from deepclustering.model import Model
from deepclustering.scheduler.customized_scheduler import RampScheduler


class TestTrainer(TestCase):

    def setUp(self) -> None:
        super().setUp()
        arch_dict = {
            "name": "clusternet6cTwoHead",
            "input_size": 24,
            "num_channel": 1,
            "output_k_A": 50,
            "output_k_B": 10,
            "num_sub_heads": 5
        }
        optim_dict = {
            "name": "Adam"
        }
        scheduler_dict = {
            "name": "MultiStepLR",
            "milestones": [10, 20, 30, 40, 50, 60, 70, 80, 90],
            "gamma": 1,
        }

        self.model = Model(arch_dict, optim_dict, scheduler_dict)
        self.scheduler = RampScheduler(10, 100, 1, -5)

    def test_save_trainer(self):
        for epoch in range(50):
            self.model.step()
            self.scheduler.step()
            print(self.scheduler.value)