import torch.nn as nn

from src.tasks.base_task import BaseTask
from src.utils import l1_reg, mmcs_fn

class AutoencoderDisentangle(BaseTask):

    def __init__(self, params):
        super().__init__(params)
        # Loss function
        self.recon_loss_fn = nn.MSELoss()
        # Task specific params
        self.reg_param = self.train_cfg['reg_param']

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, features, split):
        encoded, decoded, num_coeffs = self.forward(features)
        loss =  self.recon_loss_fn(decoded, features) +  self.reg_param * l1_reg(encoded)
        metrics = {'loss': loss}
        self.log(f'{split.capitalize()}/loss', loss, prog_bar=True)

        # Compute other metrics if relevant
        if self.compute_mmcs:
            learned_features = self.model.get_learned_features()
            dataset_features = self.train_dataset.gt
            mmcs = mmcs_fn(learned_features, dataset_features)
            metrics['mmcs'] = mmcs
            self.log(f'{split.capitalize()}/mmcs', mmcs, prog_bar=True)

        return metrics

    def training_step(self, batch, batch_nb):
        metrics = self._shared_step(batch, 'train')
        self.all_train_metrics.append(metrics)
        return metrics['loss']
