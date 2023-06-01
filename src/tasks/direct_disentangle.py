import torch
from torch.utils.data import DataLoader

from src.tasks.base_task import BaseTask
from src.utils import mmcs_fn
from src.models import ALL_MODELS

class DirectDisentangle(BaseTask):

    def __init__(self, params):
        # Add in a placeholder
        model_cfgs = params['model']['args']['activations'] = None
        super().__init__(params)

        self.model_name = self.hparams['model']['name']
        model_cfgs = self.hparams['model']['args']
        model_cfgs['activations'] = torch.from_numpy(self.train_dataset.data.T)
        self.model = ALL_MODELS[self.model_name](**model_cfgs)

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, features, split):
        loss = self.forward(features)
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

    def train_dataloader(self):
        data_loader = DataLoader(self.train_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.train_loader_worker)
        return data_loader
