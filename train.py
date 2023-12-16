import warnings
warnings.filterwarnings('ignore')

import os
from importlib import import_module
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from gcmic.config import load_cfg


def init_data(cfg):
    DATA_MODULE = import_module(cfg.data.module)
    dataloader = getattr(DATA_MODULE, cfg.data.loader)

    if cfg.general.task == "train":
        print("=> loading the train and val datasets...")
    else:
        print("=> loading the {} dataset...".format(cfg.data.split))
        
    datasets, dataloaders = dataloader(cfg)
    print("=> loading dataset completed")

    return datasets, dataloaders

def init_trainer(cfg):
    callbacks = []
    monitor = ModelCheckpoint(
        monitor="val/{}".format(cfg.model.ckpt_monitor),
        mode=cfg.model.ckpt_mode,
        dirpath=cfg.general.root,
        filename="model",
        save_last=True,
        save_on_train_epoch_end=False
    )
    callbacks.append(monitor)
    if cfg.callbacks:
        custom_callbacks = [getattr(import_module("gcmic.callbacks"), CB)() for CB in cfg.callbacks]
        callbacks.extend(custom_callbacks)

    trainer = pl.Trainer( 
        accelerator="gpu",
        devices=-1, 
        strategy='ddp_find_unused_parameters_false', 
        # strategy='ddp', 
        num_nodes=args.num_nodes,
        max_epochs=cfg.train.epochs,
        # max_steps=cfg.train.steps,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps, # validate on all val data before training 
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        callbacks=callbacks, # comment when debug
        logger=WandbLogger(name="logs", save_dir=cfg.general.root, project=cfg.general.experiment),
        profiler="simple",
    )

    return trainer

def init_model(cfg):
    if cfg.model.ckpt_path:
        print("=> configuring trainer with checkpoint from {} ...".format(cfg.model.ckpt_path))
        ckpt_path = cfg.model.ckpt_path
    elif cfg.model.use_checkpoint:
        print("=> configuring trainer with checkpoint from {} ...".format(cfg.model.use_checkpoint))
        ckpt_path = os.path.join(cfg.general.root, cfg.model.use_checkpoint)
    else:
        ckpt_path = None
    cfg.model.ckpt_path = ckpt_path
    
    MODEL = getattr(import_module(cfg.model.module), cfg.model.classname)
    model = MODEL(cfg)

    return model


if __name__ == '__main__':
    print("=> loading configurations...")
    cfg, args = load_cfg()

    print("=> initializing data...")
    datasets, dataloaders = init_data(cfg)

    print("=> initializing trainer...")
    trainer = init_trainer(cfg)
    
    print("=> initializing model...")
    model = init_model(cfg)

    if args.task == 'train':
        print("=> start training...")
        trainer.fit(model=model, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["val"], ckpt_path=cfg.model.ckpt_path)
    elif args.task == 'finetune':
        print("=> start finetuning...")
        ckpt = torch.load(cfg.model.ckpt_path)
        ckpt['epoch'] = ckpt['global_step'] = 0 # reset current_epoch and global_step
        del ckpt['callbacks'] # re-initialize ModelCheckpoint
        new_ckpt_path = os.path.join(cfg.general.root, "model.ckpt")
        torch.save(ckpt, new_ckpt_path)
        trainer.fit(model=model, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["val"], ckpt_path=new_ckpt_path)