import matplotlib
from tensorboardX import SummaryWriter as _SummaryWriter

matplotlib.use("agg")


class SummaryWriter(_SummaryWriter):
    def __init__(self, log_dir=None, comment="", **kwargs):
        assert log_dir is not None, f"log_dir should be provided, given {log_dir}."
        log_dir = str(log_dir) + "/tensorboard"
        super().__init__(log_dir, comment, **kwargs)

    def add_scalar_with_tag(
        self, tag, tag_scalar_dict, global_step=None, walltime=None
    ):
        """
        Add one-level dictionary {A:1,B:2} with tag
        :param tag: main tag like `train` or `val`
        :param tag_scalar_dict: dictionary like {A:1,B:2}
        :param global_step: epoch
        :param walltime: None
        :return:
        """
        for k, v in tag_scalar_dict.items():
            # self.add_scalars(main_tag=tag, tag_scalar_dict={k: v})
            self.add_scalar(tag=f"{tag}/{k}", scalar_value=v, global_step=global_step)
