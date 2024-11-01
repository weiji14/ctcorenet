"""
The CTCoreNet neural network model architecture and data loading modules.

Code structure adapted from Pytorch Lightning project seed at
https://github.com/PyTorchLightning/deep-learning-project-template
"""

import argparse

import lightning.pytorch as pl
import torch
import torchmetrics
import torchvision
from ctcoredata import CTCoreDataModule

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

        # self.conv_layer1 = torch.nn.Conv2d(
        #     in_channels=1, out_channels=3, kernel_size=3, padding=1  # 'same' padding
        # )

        # Unet model
        # https://github.com/mateuszbuda/brain-segmentation-pytorch#pytorch-hub
        self.unet = torch.hub.load(
            repo_or_dir="mateuszbuda/brain-segmentation-pytorch:v1.0",
            model="unet",
            in_channels=1,
            out_channels=1,  # binary classification
            init_features=32,
            pretrained=False,
            trust_repo="check",
        )

        # self.output_conv = torch.nn.Conv2d(32, 1, 1, 1)

        # Intersection over Union (IoU) metric
        self.iou = torchmetrics.JaccardIndex(task="binary", num_classes=2)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass (Inference/Prediction).
        """
        # Pass tensor through Convolutional Layers
        # x = self.conv_layer1(x)
        # x = F.leaky_relu(x)

        # Pass tensors through Unet
        x = self.unet(x)

        # # Final output convolution
        # x = self.output_conv(x)

        return x

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Logic for the neural network's training loop.

        For each batch:
        1. Get the image and corresponding groundtruth label from each batch
        2. Pass the image through the neural network to get a predicted label
        3. Calculate the loss between the predicted label and groundtruth label

        Using the focal loss from RetinaNet, with tunable parameters alpha and
        gamma.

        Reference:
        - Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2018).
          Focal Loss for Dense Object Detection. ArXiv:1708.02002 [Cs].
          https://arxiv.org/abs/1708.02002
        """
        # image, label = batch  # x is the tensor, y is the label
        image: torch.Tensor = batch[:, 0:1, :, :]  # Input CT Core image
        label: torch.Tensor = batch[:, 1:2, :, :]  # Classified pixel labels

        logits: torch.Tensor = self(image)  # pass image through neural network
        loss: torch.Tensor = torchvision.ops.sigmoid_focal_loss(
            inputs=logits,
            targets=label,
            alpha=0.75,  # 0.5
            gamma=2,  # 7
            reduction="mean",
        )
        iou = self.iou(preds=logits, target=label.to(dtype=torch.uint8))

        # Log loss value and images to Tensorboard
        self.logger.experiment.add_scalar(
            tag="Loss", scalar_value=loss, global_step=self.global_step
        )
        self.logger.experiment.add_scalar(
            tag="IoU", scalar_value=iou, global_step=self.global_step
        )
        logit_grid: torch.Tensor = torchvision.utils.make_grid(tensor=logits)
        self.logger.experiment.add_image(
            tag="logit", img_tensor=logit_grid, global_step=self.global_step
        )
        if self.global_step == 0:
            label_grid: torch.Tensor = torchvision.utils.make_grid(tensor=label)
            self.logger.experiment.add_image(
                tag="label", img_tensor=label_grid, global_step=self.global_step
            )
            image_grid: torch.Tensor = torchvision.utils.make_grid(tensor=image)
            self.logger.experiment.add_image(
                tag="image", img_tensor=image_grid, global_step=self.global_step
            )

        return loss

    def configure_optimizers(self):
        """
        Optimizing function used to reduce the loss, so that the predicted
        label gets as close as possible to the groundtruth label.

        Using the Adam optimizer with a learning rate of 0.001. See:

        - Kingma, D. P., & Ba, J. (2017). Adam: A Method for Stochastic
          Optimization. ArXiv:1412.6980 [Cs]. http://arxiv.org/abs/1412.6980
        """
        return torch.optim.Adam(params=self.parameters(), lr=0.01)


def cli_main(hparams):
    """
    Command line interface to run the CTCoreNet model. Based on
    https://github.com/PyTorchLightning/deep-learning-project-template

    The script can be executed in the terminal using:

        python ctcorenet/ctcoreunet.py

    This will 1) load the data, 2) initialize the model, 3) train the model,
    4) test the model, and 5) export the model.

    To train the model for 3 epochs on a single GPU, use the following command:

        python ctcorenet/ctcoreunet.py --max_epochs=3 --gpus=1

    To train the model on 2 GPUs using Distributed Data Parallel (DDP)
    processing, with CUDNN deterministic and floating point 16 mode enabled,
    try doing something like this:

        python ctcorenet/ctcoreunet.py --max_epochs=3 --gpus=2
               --accelerator=ddp --deterministic=True --precision=16

    More options can be found by using `python ctcorenet/ctcoreunet.py --help`.
    Happy training!
    """
    # Set a seed to control for randomness
    pl.seed_everything(42)

    # Load Data
    ctcoredatamodule = CTCoreDataModule()

    # Initialize Model
    model = CTCoreNet()

    # Training
    trainer = pl.Trainer(
        accelerator=hparams.accelerator,
        deterministic=hparams.deterministic,
        max_epochs=hparams.max_epochs,
        devices=hparams.devices,
        precision=hparams.precision,
    )
    trainer.fit(model=model, datamodule=ctcoredatamodule)

    # Export Model to ONNX format
    model.to_onnx(
        file_path="ctcorenet/ctcorenet_model.onnx",
        input_sample=torch.randn(1, 1, 512, 512),
        export_params=False,
        opset_version=11,
    )


if __name__ == "__main__":
    ## Parse arguments from command-line
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default="auto")  # "gpu" or "mps" (mac M1 chip)
    parser.add_argument("--deterministic", default=None)
    parser.add_argument("--max_epochs", default=3, type=int)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default=None)
    args = parser.parse_args()

    cli_main(args)
