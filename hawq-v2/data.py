"""
Tiny-ImageNet data loading for HAWQ-v2 experiments.

The HuggingFace dataset `zh-plus/tiny-imagenet` has 200 classes with 500
training images and 50 validation images per class. Each item is a PIL
image plus an integer label in [0, 200). Images are 64x64 - we resize to
224x224 to match what DeiT expects.

Returned DataLoaders yield (tensor, label_int) tuples directly compatible
with the rest of the pipeline.
"""

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Standard ImageNet preprocessing stats - these match what DeiT was
# pretrained with, so reusing them is the right thing to do.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TinyImageNetHF(Dataset):
    """Thin adapter that turns a HuggingFace split into (tensor, int) pairs."""

    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        label = int(item["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size=224, train=False):
    """
    Eval transform: deterministic resize + ImageNet normalization.
    Train transform: light augmentation (random crop + horizontal flip).
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _pick_split(ds, *candidates):
    """Return the first split key in `ds` that matches one of `candidates`."""
    for c in candidates:
        if c in ds:
            return c
    raise KeyError(
        f"Could not find any of splits {candidates} in dataset; "
        f"available splits: {list(ds.keys())}"
    )


def load_tiny_imagenet(
    batch_size=64,
    num_workers=2,
    image_size=224,
    train_subset=None,
    val_subset=None,
    augment_train=True,
):
    """
    Returns
    -------
    train_loader, val_loader, num_classes
    """
    from datasets import load_dataset

    ds = load_dataset("zh-plus/tiny-imagenet")

    train_key = _pick_split(ds, "train")
    val_key = _pick_split(ds, "valid", "validation", "test")

    train_split = ds[train_key]
    val_split = ds[val_key]

    if train_subset is not None:
        train_split = train_split.select(range(min(int(train_subset), len(train_split))))
    if val_subset is not None:
        val_split = val_split.select(range(min(int(val_subset), len(val_split))))

    train_tf = build_transforms(image_size=image_size, train=augment_train)
    eval_tf = build_transforms(image_size=image_size, train=False)

    train_ds = TinyImageNetHF(train_split, transform=train_tf)
    val_ds = TinyImageNetHF(val_split, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, 200
