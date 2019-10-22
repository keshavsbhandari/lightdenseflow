import torch
from torch.nn import functional as F
from models.Flowestimator import FlowEstimator
import pytorch_lightning as pl
from utils import warper
from dataloader.sintelloader import SintelLoader
from utils.photometricloss import photometricloss
from torch.optim.lr_scheduler import ReduceLROnPlateau


class FlowTrainer(object):
    def __init__(self):
        super(FlowTrainer, self).__init__()
        # not the best model...
        self.model = FlowEstimator()
        self.optimizer = None
        self.lr_scheduler = None
        self.save_dir = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.tensorboardlogs_path = None
        self.save_model_path = None
        self.load_model_path = None
        self.best_metrics = None
        self.gpu_ids = None

    def initialize(self):
        self.model.to()
    def forward(self, x):
        return self.deepflow(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        flow, occlusion = self.forward(batch['displacement'])
        loss = photometricloss(batch, flow, occlusion)
        tensorboard_logs = {'photometric_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        flow, occlusion = self.forward(batch['displacement'])
        loss = photometricloss(batch, flow, occlusion)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer,],[scheduler,]

    def train_dataloader(self):
        # REQUIRED
        return SintelLoader(batch_size=10).load()

    def val_dataloader(self):
        # OPTIONAL
        return SintelLoader(batch_size=10).load()

    def test_dataloader(self):
        # OPTIONAL
        return SintelLoader(batch_size=10).load()
