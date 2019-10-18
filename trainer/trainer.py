import torch
from torch.nn import functional as F
from models.flowestimator import FlowEstimator
import pytorch_lightning as pl
from dataloader.sintelloader import  SintelLoader


class CoolSystem(pl.LightningModule):
    def __init__(self):
        super(CoolSystem, self).__init__()
        # not the best model...
        self.deepflow = FlowEstimator()

    def forward(self, x):
        return self.deepflow(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x = batch['displacement']
        flow = self.forward(x)
        loss = F.mse_loss(flow, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return SintelLoader(batch_size=10).load()

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return SintelLoader(batch_size=10).load()

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return SintelLoader(batch_size=10).load()
