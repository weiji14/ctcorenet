"""
The CTCoreNet neural network's data loading modules.

Code structure adapted from Pytorch Lightning project seed at
https://github.com/PyTorchLightning/deep-learning-project-template
"""
import glob
import os
import typing

import pytorch_lightning as pl
import torch
import torchvision
from torch.nn import functional as F


class CTCoreDataset(torch.utils.data.Dataset):
    """
    Training image data and groundtruth labels for the CTCoreNet project.
    Subclassed from
    https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    CT core images and label masks are loaded from PNG/JPG files using
    torchvision.

    Parameters
    ----------
    images_dir : str
        Filepath to the folder containing the CT core images. Default is
        'data/train/'.
    """

    def __init__(self, images_dir: str = "data/train/", predict: bool = False):
        self.images_dir: str = images_dir
        self.predict: bool = predict

        pathname: str = (
            os.path.join(self.images_dir, "*.jpg")
            if self.predict
            else os.path.join(self.images_dir, "**", "img.png")
        )
        self.image_paths: typing.List[str] = sorted(glob.glob(pathname=pathname))

        # List of raw JPG images of CT Core and classified pixel labels
        self.images_and_labels: typing.List[torch.Tensor] = []
        # List of raw image shapes (channel, height, width) before resizing
        self.original_shapes: typing.List[tuple] = []

        # Generate training dataset with augmentation
        for img_path in self.image_paths:
            # Load image from file
            image: torch.Tensor = torchvision.io.read_image(
                path=img_path, mode=torchvision.io.ImageReadMode.GRAY
            )

            # Prediction mode - Load only CT Core images
            if self.predict:
                # Resize image to standard shape so that they can be stacked
                original_shape: tuple = image.shape
                self.original_shapes.append(tuple(original_shape))

                tensor: torch.Tensor = torch.stack(tensors=[image]).float()
                _image: torch.Tensor = F.interpolate(
                    input=tensor, size=(4096, 512), mode="bilinear", align_corners=True
                )
                self.images_and_labels.append(_image[0])

            # Training mode - Load CT Core images and Groundtruth labels
            else:
                label: torch.Tensor = torchvision.io.read_image(
                    path=os.path.join(os.path.dirname(img_path), "label.png")
                )
                assert image.shape == label.shape
                image_and_label = torch.cat(tensors=[image, label])
                # print(image_and_label.shape)

                # Ensure standard tensor size
                five_crop_transform = torchvision.transforms.FiveCrop(size=(256, 256))
                self.images_and_labels.extend(five_crop_transform(image_and_label))
                random_crop_transform = torchvision.transforms.RandomCrop(
                    size=(256, 256), padding_mode="symmetric"
                )
                self.images_and_labels.extend(
                    [random_crop_transform(image_and_label) for _ in range(123)]
                )

        self.ids = [i for i in range(len(self.images_and_labels))]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images_and_labels[index].float()

    def __len__(self) -> int:
        return len(self.ids)


class CTCoreDataModule(pl.LightningDataModule):
    """
    Data preparation code to load the CT Core image data into Python.
    Specifically, sediment cores from Ross Sea, Antarctica drilled in 2015
    RS15-LC42 and RS15-LC48.

    References:
    - https://doi.org/10.1594/PANGAEA.920653
    - https://doi.org/doi:10.22663/KOPRI-KPDC-00000518.1

    This is a reusable Pytorch Lightning Data Module with a custom Dataset. See
    https://pytorch-lightning.readthedocs.io/en/1.4.1/extensions/datamodules.html
    and https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

    Parameters
    ----------
    images_dir : str
        Filepath to the folder containing the CT Core images. Default is
        'data/train/'.

    predict : bool
        Whether the data module is being used for prediction on new images, in
        which case no groundtruth labels are to be loaded. Default is False,
        i.e. groundtruth labels will be loaded from a label.png file.
    """

    def __init__(self, images_dir: str = "data/train/", predict: bool = False):
        """
        Define image data storage location and data containers for storing
        preprocessed images and labels.
        """
        super().__init__()
        self.images_dir: str = images_dir
        self.predict: bool = predict

    def prepare_data(self):
        """
        Data operations to perform on a single CPU.
        Load image data and labels from folders, do preprocessing, etc.
        """
        # Create a proper Pytorch Dataset from tuple of (images, targets)
        self.dataset: torch.utils.data.Dataset = CTCoreDataset(
            images_dir=self.images_dir, predict=self.predict
        )

    def setup(self, stage: typing.Optional[str] = None) -> torch.utils.data.Dataset:
        """
        Data operations to perform on every GPU.
        Split data into training and test sets, etc.
        """
        return self.dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the training loop.
        Set the training batch size here too.
        """
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the prediction loop.
        Set the prediction batch size here too.
        """
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_size=8)
