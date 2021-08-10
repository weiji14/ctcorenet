"""
The CTCoreNet neural network model architecture and data loading modules.

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

torch.use_deterministic_algorithms(mode=True)


class CTCoreNet(pl.LightningModule):
    """
    Neural network for classifying rock clasts in computed tomography (CT)
    scans of sediment cores.

    Implemented using Pytorch Lightning.
    """

    def __init__(self):
        """
        Define layers of the Convolutional Neural Network.
        """
        super().__init__()

        self.conv_layer1 = torch.nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1  # 'same' padding
        )
        self.output_conv = torch.nn.Conv2d(32, 1, 1, 1)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass (Inference/Prediction).
        """
        # Pass tensor through Convolutional Layers
        x = self.conv_layer1(x)
        x = F.leaky_relu(x)

        # Final output convolution
        x = self.output_conv(x)
        return x

    def training_step(self, batch, batch_idx) -> float:
        """
        Logic for the neural network's training loop.

        For each batch:
        1. Get the image and corresponding groundtruth label from each batch
        2. Pass the image through the neural network to get a predicted label
        3. Calculate the loss between the predicted label and groundtruth label
        """
        image = batch[0][:, 0:1, :, :].float()  # Input CT Core image
        label = batch[0][:, 1:2, :, :].float()  # Classified pixel labels

        logits = self(image)  # pass the image through the neural network model
        loss = F.binary_cross_entropy_with_logits(input=logits, target=label)
        return loss

    def configure_optimizers(self):
        """
        Optimizing function used to reduce the loss, so that the predicted
        label gets as close as possible to the groundtruth label.

        Using the Adam optimizer with a learning rate of 0.001. See:

        - Kingma, D. P., & Ba, J. (2017). Adam: A Method for Stochastic
          Optimization. ArXiv:1412.6980 [Cs]. http://arxiv.org/abs/1412.6980
        """
        return torch.optim.Adam(params=self.parameters(), lr=0.001)


class CTCoreData(pl.LightningDataModule):
    """
    Data preparation code to load the CT Core image data into Python.

    This is a reusable Pytorch Lightning Data Module with a custom Dataset. See
    https://pytorch-lightning.readthedocs.io/en/1.4.1/extensions/datamodules.html
    and https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """

    def __init__(self, images_dir: str = "data/train/"):
        """
        Define image data storage location and data containers for storing
        preprocessed images and labels.
        """
        super().__init__()
        self.images_dir = images_dir
        # list of raw JPG images of CT Core and classified pixel labels
        self.images_and_labels = []

    def prepare_data(self):
        """
        Data operations to perform on a single CPU.
        Load image data and labels from folders, do preprocessing, etc.
        """

    def setup(self, stage: typing.Optional[str] = None) -> torch.utils.data.Dataset:
        """
        Data operations to perform on every GPU.
        Split data into training and test sets, etc.
        """
        for img_path in glob.glob(os.path.join(self.images_dir, "**", "img.png")):
            # Load image and label from file
            image: torch.Tensor = torchvision.io.read_image(
                path=img_path, mode=torchvision.io.ImageReadMode.GRAY
            )
            label: torch.Tensor = torchvision.io.read_image(
                path=os.path.join(os.path.dirname(img_path), "label.png")
            )
            assert image.shape == label.shape
            image_and_label = torch.cat(tensors=[image, label])
            # print(image_and_label.shape)

            # Ensure standard tensor size
            crop_transform = torchvision.transforms.FiveCrop(size=(256, 256))
            self.images_and_labels.extend(crop_transform(image_and_label))

        # Create a proper Pytorch Dataset from tuple of (image, label)
        self.dataset = torch.utils.data.TensorDataset(
            torch.stack(tensors=self.images_and_labels)
        )
        return self.dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the training loop.
        Set the training batch size here too.
        """
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_size=2)


def cli_main():
    """
    Command line interface to run the CTCoreNet model. Based on
    https://github.com/PyTorchLightning/deep-learning-project-template

    The script can be executed in the terminal using:

        python ctcorenet/ctcorenet.py

    This will 1) load the data, 2) initialize the model, 3) train the model,
    and 4) test the model.
    """
    # Set a seed to control for randomness
    pl.seed_everything(42)

    # Load Data
    ctcoredatamodule = CTCoreData()

    # Initialize Model
    model = CTCoreNet()

    # Training
    trainer = pl.Trainer(precision=16, gpus=1, max_epochs=3)  # GPU
    # trainer = pl.Trainer(max_epochs=3) # CPU
    trainer.fit(model=model, datamodule=ctcoredatamodule)


if __name__ == "__main__":
    cli_main()
