"""
LightningModule for training and validating a segmentation model using the
Segmentor class.
"""

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import F1Score, MulticlassJaccardIndex, BinaryJaccardIndex

from finetune.segment.factory import Segmentor
from finetune.segment.joinLoss import JointLoss


class ChesapeakeSegmentor(L.LightningModule):
    """
    LightningModule for segmentation tasks, utilizing Clay Segmentor.

    Attributes:
        model (nn.Module): Clay Segmentor model.
        loss_fn (nn.Module): The loss function.
        iou (Metric): Intersection over Union metric.
        f1 (Metric): F1 Score metric.
        lr (float): Learning rate.
    """

    def __init__(  # # noqa: PLR0913
        self,
        num_classes,
        ckpt_path,
        lr,
        wd,
        b1,
        b2,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing
        self.model = Segmentor(
            num_classes=num_classes,
            ckpt_path=ckpt_path,
        )

        #self.loss_fn = smp.losses.FocalLoss(mode="multiclass")
        self.focal_loss = smp.losses.FocalLoss(mode="binary")
        self.dice_loss = smp.losses.DiceLoss(mode="binary")
        self.loss_fn = JointLoss(self.focal_loss, self.dice_loss, weight1=0.5, weight2=0.5)

        #self.iou = MulticlassJaccardIndex(
        #    num_classes=num_classes,
        #    average="weighted",
        #)
        #self.f1 = F1Score(
        #    task="multiclass",
        #    num_classes=num_classes,
        #    average="weighted",
        #)

        self.iou = BinaryJaccardIndex()
        self.f1 = F1Score(task='binary')

    def forward(self, datacube):
        """
        Forward pass through the segmentation model.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The segmentation logits.
        """

        #waves = torch.tensor([0.65, 0.56, 0.48, 0.842])  # NAIP wavelengths
        #gsd = torch.tensor(1.0)  # NAIP GSD
        #waves = torch.tensor([0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19])  # sentinel-2-l2a wavelengths
        #gsd = torch.tensor(10.0)  # sentinel-2-l2a
        waves = torch.tensor([0.65, 0.56, 0.48, 0.86, 1.6, 2.2])  # landsat-c2l2-sr wavelengths
        gsd = torch.tensor(30.0)  # landsat-c2l2-sr

        # Forward pass through the network
        return self.model(
            {
                "pixels": datacube["pixels"],
                "time": datacube["time"],
                "latlon": datacube["latlon"],
                "gsd": gsd,
                "waves": waves,
            },
        )

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler
            configuration.
        """
        optimizer = optim.AdamW(
            [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=1,
            eta_min=self.hparams.lr * 100,
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training and validation.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.
            phase (str): The phase (train or val).

        Returns:
            torch.Tensor: The loss value.
        """
        labels = batch["label"].long()
        outputs = self(batch)
        #print(outputs.shape)
        if outputs.shape[-2:] != (256, 256):
            outputs = F.interpolate(
                outputs,
                size=(256, 256),#size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )  # Resize to match labels size

        loss = self.loss_fn(outputs, labels)
        iou = self.iou(outputs, labels)
        f1 = self.f1(outputs, labels)

        # Log metrics
        self.log(
            f"{phase}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/iou",
            iou,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/f1",
            f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "val")
