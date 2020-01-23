from unittest import TestCase
from deepclustering.dataset import ACDCDataset
from deepclustering.dataset import PatientSampler
from deepclustering import DATA_PATH
from torch.utils.data import DataLoader
from deepclustering.dataloader.sampler import InfiniteRandomSampler


class TestPatientSampler(TestCase):

    def test_acdc_sampler(self):
        dataset = ACDCDataset(root_dir=DATA_PATH, mode="train", subfolders=["img", "gt"])
        patient_sampler = PatientSampler(dataset=dataset, grp_regex=dataset._pattern, shuffle=True,
                                         infinite_sampler=True)
        dataloader = DataLoader(dataset, batch_sampler=patient_sampler)
        for _, filename in dataloader:
            print(filename)
