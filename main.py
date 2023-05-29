from pathlib import Path
import yaml

import fire
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from src.tasks import ALL_TASKS

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def seed_run(seed):
    if seed is None:
        print(f"No seed provided; running experiment without seed")
        return
    print(f"Seeding run with: {seed}")
    pl.seed_everything(seed)

def train(cfg_path):
    cfg = read_yaml(cfg_path)
    trainer_cfg = cfg['training']

    random_seed = cfg.get('random_seed')
    seed_run(random_seed)

    # GPU
    accelerator = trainer_cfg.get('accelerator', 'cpu')
    devices = trainer_cfg.get('devices', 'auto')

    # Logging
    save_dir = trainer_cfg.get('save_dir', '/tmp/disentangle')
    exp_name = cfg['exp_name']
    version = cfg['version']
    wandb_entity = cfg['wandb_entity']
    wandb_project = cfg['wandb_project']
    #logger = WandbLogger(name=f'{exp_name}_v{version}', save_dir=save_dir,
    #                     project=wandb_project, entity=wandb_entity)

    # Callbacks - Checkpointing and early stop
    save_top_k = trainer_cfg.get('save_top_k', 2)
    monitor_metric = trainer_cfg.get('monitor_metric', 'Eval_Loss')
    monitor_mode = trainer_cfg.get('monitor_mode', 'min')
    patience = trainer_cfg.get('patience', 10)

    ckpt_dir = Path(save_dir) / exp_name / f'version_{version}' / 'ckpt'
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=save_top_k, 
                              verbose=True, monitor=monitor_metric, 
                              mode=monitor_mode, every_n_epochs=1)
    earlystop_cb = EarlyStopping(monitor=monitor_metric, 
                                 patience=patience, 
                                 verbose=True, mode=monitor_mode)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Trainer config
    limit_train_batches = trainer_cfg.get('limit_train_batches', 1.0)
    enable_model_summary = trainer_cfg.get('enable_model_summary', False)
    max_epochs = trainer_cfg.get('max_epochs', 100)

    trainer = Trainer(accelerator=accelerator,
                      devices=devices,
                      #logger=logger,
                      callbacks=[ckpt_cb, earlystop_cb, lr_monitor],
                      limit_train_batches=limit_train_batches,
                      enable_model_summary = enable_model_summary,
                      max_epochs=max_epochs,
                      log_every_n_steps=20)
    
    # Setup the task
    task_name = cfg['task']
    task = ALL_TASKS[task_name](cfg)
    trainer.fit(task)


if __name__ == "__main__":
    fire.Fire()