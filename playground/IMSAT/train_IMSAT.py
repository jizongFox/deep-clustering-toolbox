from torchvision.transforms import Compose, RandomCrop, ToTensor

from deepclustering.manager import ConfigManger
from deepclustering.model import Model
from deepclustering.utils import fix_all_seed
from playground.IMSAT.IMSATTrainer import IMSATTrainer
from playground.IMSAT.mnist_helper import MNISTClusteringDatasetInterface

fix_all_seed(3)

DEFAULT_CONFIG = "./IMSAT.yaml"

merged_config = ConfigManger(
    DEFAULT_CONFIG_PATH=DEFAULT_CONFIG, verbose=True, integrality_check=True
).config

tf1 = Compose([ToTensor()])
tf2 = Compose([RandomCrop((28, 28), padding=2), ToTensor()])

# create model:
model = Model(
    arch_dict=merged_config["Arch"],
    optim_dict=merged_config["Optim"],
    scheduler_dict=merged_config["Scheduler"],
)

train_loader_A = MNISTClusteringDatasetInterface(
    **merged_config["DataLoader"]
).ParallelDataLoader(tf1, tf2, tf2, tf2, tf2)

val_loader = MNISTClusteringDatasetInterface(
    **merged_config["DataLoader"]
).ParallelDataLoader(tf1)

trainer = IMSATTrainer(
    model=model,
    train_loader=train_loader_A,
    val_loader=val_loader,
    config=merged_config,
    **merged_config["Trainer"]
)
trainer.start_training()
trainer.clean_up()
