from ._hooks import HookBase
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from typing import List, Union, Callable, Dict
from deepclustering.utils import tqdm_
from deepclustering.trainer import _Trainer
from deepclustering.utils import flatten_dict, filter_dict, nice_dict
import sys
from termcolor import colored


class TQDMTaskBar(HookBase):
    _trainer: _Trainer

    def __init__(
        self,
        train_batches: int,
        val_batches: int,
        train_groupname="train",
        val_groupname="val",
    ) -> None:
        super().__init__()

        self._train_batches = train_batches
        self._val_batches = val_batches
        self._train_groupname = train_groupname
        self._val_groupname = val_groupname

    def before_train_epoch(self, *args, **kwargs):
        self.tqdm_indicator = tqdm_(
            range(self._train_batches), total=self._train_batches
        )
        self._epoch = kwargs.get("epoch")
        if self._epoch is not None:
            self.tqdm_indicator.set_description(f"  Training Epoch {self._epoch}")

    def before_eval_epoch(self, *args, **kwargs):
        self.tqdm_indicator = tqdm_(range(self._val_batches), total=self._val_batches)
        self._epoch = kwargs.get("epoch")
        if self._epoch is not None:
            self.tqdm_indicator.set_description(f"Evaluating Epoch {self._epoch}")

    def after_train_step(self, *args, **kwargs):
        self.report_dict = filter_dict(
            flatten_dict(
                self._trainer._meter_interface.tracking_status(
                    group_name=self._train_groupname
                )
            )
        )
        self.tqdm_indicator.update()
        self.tqdm_indicator.set_postfix(self.report_dict)

    def after_eval_step(self, *args, **kwargs):
        self.report_dict = filter_dict(
            flatten_dict(
                self._trainer._meter_interface.tracking_status(
                    group_name=self._val_groupname
                )
            )
        )
        self.tqdm_indicator.update()
        self.tqdm_indicator.set_postfix(self.report_dict)

    def after_train_epoch(self, *args, **kwargs):
        print(
            colored(
                f"  Training Epoch {self._epoch}: {nice_dict(self.report_dict)}",
                "green",
            ),
            flush=True,
        )

    def after_eval_epoch(self, *args, **kwargs):
        print(
            colored(
                f"Evaluating Epoch {self._epoch}: {nice_dict(self.report_dict)}", "red"
            ),
            flush=True,
        )
