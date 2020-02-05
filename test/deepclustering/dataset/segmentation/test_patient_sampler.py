from unittest import TestCase
from deepclustering.dataset import ACDCDataset
from deepclustering.dataset import PatientSampler
from deepclustering import DATA_PATH
from torch.utils.data import DataLoader
from deepclustering.dataloader.sampler import InfiniteRandomSampler


class TestPatientSampler(TestCase):
    def test_acdc_sampler(self):
        dataset = ACDCDataset(root_dir="./", mode="train", subfolders=["img", "gt"])
        patient_sampler = PatientSampler(
            dataset=dataset,
            grp_regex=dataset._pattern,
            shuffle=True,
            infinite_sampler=True,
        )
        dataloader = DataLoader(dataset, batch_sampler=patient_sampler)
        for i, (_, filename) in enumerate(dataloader):
            print(filename)
            if i == 100:
                break
