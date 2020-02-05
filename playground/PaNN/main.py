from deepclustering.augment import SequentialWrapper, pil_augment
from deepclustering.dataset.segmentation.acdc_dataset import ACDCSemiInterface
from deepclustering.manager import ConfigManger
from deepclustering.model import Model
from deepclustering.utils import fix_all_seed
from playground.PaNN.trainer import SemiSegTrainer

config = ConfigManger(
    DEFAULT_CONFIG_PATH="config.yaml", integrality_check=True, verbose=True
).merged_config

fix_all_seed(config.get("Seed", 0))

data_handler = ACDCSemiInterface(labeled_data_ratio=0.99, unlabeled_data_ratio=0.01)
data_handler.compile_dataloader_params(
    labeled_batch_size=4,
    unlabeled_batch_size=8,
    val_batch_size=1,
    shuffle=True,
    num_workers=2,
)
# transformations
train_transforms = SequentialWrapper(
    img_transform=pil_augment.Compose(
        [
            pil_augment.Resize((256, 256)),
            pil_augment.RandomCrop((224, 224)),
            pil_augment.RandomHorizontalFlip(),
            pil_augment.RandomRotation(degrees=10),
            pil_augment.ToTensor(),
        ]
    ),
    target_transform=pil_augment.Compose(
        [
            pil_augment.Resize((256, 256)),
            pil_augment.RandomCrop((224, 224)),
            pil_augment.RandomHorizontalFlip(),
            pil_augment.RandomRotation(degrees=10),
            pil_augment.ToLabel(),
        ]
    ),
    if_is_target=(False, True),
)
val_transform = SequentialWrapper(
    img_transform=pil_augment.Compose(
        [pil_augment.CenterCrop((224, 224)), pil_augment.ToTensor()]
    ),
    target_transform=pil_augment.Compose(
        [pil_augment.CenterCrop((224, 224)), pil_augment.ToLabel()]
    ),
    if_is_target=(False, True),
)
labeled_loader, unlabeled_loader, val_loader = data_handler.SemiSupervisedDataLoaders(
    labeled_transform=train_transforms,
    unlabeled_transform=train_transforms,
    val_transform=val_transform,
    group_labeled=True,
    group_unlabeled=False,
    group_val=True,
)
model = Model(
    arch_dict=config["Arch"],
    optim_dict=config["Optim"],
    scheduler_dict=config["Scheduler"],
)
trainer = SemiSegTrainer(
    model=model,
    labeled_loader=labeled_loader,
    unlabeled_loader=unlabeled_loader,
    val_loader=val_loader,
    config=config,
    **config["Trainer"]
)
trainer.start_training()
