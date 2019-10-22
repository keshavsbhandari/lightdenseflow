from trainer.trainer import FlowTrainer
from pytorch_lightning import Trainer

model = FlowTrainer()

trainer = Trainer(gpus = 1,distributed_backend='dp',max_nb_epochs=1, train_percent_check=0.1)
trainer.fit(model)