import matplotlib
from tensorboardX import SummaryWriter as _SummaryWriter

matplotlib.use('agg')


class SummaryWriter(_SummaryWriter):
    def __init__(self, log_dir=None, comment='', **kwargs):
        assert log_dir is not None, f'log_dir should be provided, given {log_dir}.'
        log_dir = str(log_dir) + '/tensorboard'
        super().__init__(log_dir, comment, **kwargs)
