
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import ALL_DATA
from src.models import ALL_MODELS


class BaseTask(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        # Load train configs
        self.train_cfg = self.hparams["training"]
        self.batch_size = self.train_cfg["batch_size"]
        self.train_loader_worker = self.train_cfg["train_loader_worker"]

        # Init & setup model
        self.model_name = self.hparams['model']['name']
        model_cfgs = self.hparams['model']['args']
        self.model = ALL_MODELS[self.model_name](**model_cfgs)
        
        # Setup dataset
        dataset_name = self.hparams['dataset']['name']
        dataset_params = self.hparams['dataset']['train_args']
        dataset = ALL_DATA[dataset_name](**dataset_params)
        self.train_dataset = dataset

        # Metric eval
        self.compute_mmcs = self.train_cfg["compute_mmcs"]

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_nb):
        raise NotImplementedError
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.all_train_metrics = []

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        metric_keys = list(self.all_train_metrics[0].keys())
        for metric_key in metric_keys:
            avg_val = sum(batch[metric_key] for batch in self.all_train_metrics) / len(self.all_train_metrics)
            tag = 'Train/epoch_avg_' + metric_key
            self.log(tag, avg_val, logger=True, sync_dist=True)

    def configure_optimizers(self):
        opt_name = self.train_cfg['optimizer']
        opt_cfg = self.train_cfg['optimizer_cfg']
        if 'lr' not in opt_cfg:
            raise KeyError("You must provide learning rate in optimizer cfg")
        
        if opt_name == 'Adam':
            optimizer_class = optim.Adam
        elif opt_name == "SGD":
            optimizer_class = optim.SGD
        elif opt_name == "AdamW":
            optimizer_class = optim.AdamW
        else:
            raise ValueError(f"{opt_name} is not supported, add it to configure_optimizers in base lightning class.")
        optimizer = optimizer_class(self.parameters(), **opt_cfg)
        return optimizer

    def train_dataloader(self):
        data_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.train_loader_worker)
        return data_loader


    #def configure_optimizers(self):
    #    """ Configure optimizer and scheduler """
    #    optimizer = super().configure_optimizers()
    #    self.scheduler_name = self.training_cfg.get("scheduler", None)
    #    scheduler_cfg = self.training_cfg.get("scheduler_cfg", {})

    #    # Only ReduceLROnPlateau operates on epoch interval
    #    if self.scheduler_name == "Plateau":
    #        monitor_metric = scheduler_cfg.pop("monitor_metric", "Val/epoch_avg_loss")
    #        scheduler_class = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_cfg)
    #        return [optimizer], [{"scheduler": scheduler_class, 
    #                              "monitor": monitor_metric}]
    #    elif self.scheduler_name == "MultiStep":
    #        scheduler_class = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_cfg)
    #        return [optimizer], [{"scheduler": scheduler_class,
    #                              "interval": "epoch"}]
    #    # Manually schedule for Cosine and Polynomial scheduler
    #    elif self.scheduler_name in [None, "Cosine", "Poly"]:
    #        return optimizer
    #    else:
    #        raise ValueError(f"{self.scheduler_name} is not supported, add it to configure_optimizers in BaseDownstreamTask.")


    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, 
    #                         on_tpu=False, using_native_amp=False, using_lbfgs=False):
    #    """Overwrite with warmup epoch and manual lr decay"""

    #    self.epoch_progress = self.current_epoch + min((batch_idx+1)/self.num_steps_per_train_epoch , 1)
    #    initial_lr = self.training_cfg["optimizer_cfg"]['lr']
    #    warm_up_epoch = self.training_cfg.get("warm_up_epoch", 0)
    #    max_epochs = self.training_cfg['trainer']['max_epochs']

    #    if self.scheduler_name in ["Cosine", "Poly"] or self.epoch_progress <= warm_up_epoch:
    #        if self.epoch_progress <= warm_up_epoch:
    #            lr = initial_lr * self.epoch_progress / warm_up_epoch
    #        elif self.scheduler_name == "Cosine":
    #            lr = initial_lr * 0.5 * (1. + math.cos(math.pi * (self.epoch_progress - warm_up_epoch) / (max_epochs - warm_up_epoch)))
    #        else:
    #            power = self.training_cfg.get("scheduler_cfg", {}).get("power", 0.5)
    #            lr = initial_lr * (1. - (self.epoch_progress - warm_up_epoch) / (max_epochs - warm_up_epoch)) ** power
    #        for param_group in optimizer.param_groups:
    #            param_group['lr'] = lr    

    #    optimizer.step(closure=optimizer_closure)

