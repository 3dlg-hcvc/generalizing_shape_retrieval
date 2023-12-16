import warnings
warnings.filterwarnings('ignore')

import os
from importlib import import_module
import pytorch_lightning as pl
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
    if cfg.callbacks:
        callbacks = [getattr(import_module("wizard.callbacks"), CB)() for CB in cfg.callbacks]

    trainer = pl.Trainer(
        gpus=-1, # use all available GPUs 
        strategy='ddp', # use multiple GPUs on the same machine
        log_every_n_steps=cfg.train.log_every_n_steps,
        callbacks=callbacks, # comment when debug
    )

    return trainer

def init_model(cfg):
    if cfg.model.use_checkpoint:
        exp_root = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.general.experiment)
        task_dirs = os.listdir(exp_root)
        if 'train' in task_dirs:
            ckpt_path = os.path.join(exp_root, f"train/{cfg.model.use_checkpoint}")
        else:
            ckpt_path = os.path.join(exp_root, f"finetune/{cfg.model.use_checkpoint}")
        print("=> configuring trainer with checkpoint from {} ...".format(cfg.model.use_checkpoint))
    else:
        ckpt_path = None
    cfg.model.ckpt_path = ckpt_path
    
    MODEL = getattr(import_module(cfg.model.module), cfg.model.classname)
    model = MODEL(cfg)

    return model


if __name__ == '__main__':
    print("=> loading configurations...")
    cfg, args = load_cfg()

    print("=> initializing trainer...")
    trainer = init_trainer(cfg)
    
    print("=> initializing model...")
    model = init_model(cfg)
    
    if args.task == 'test':
        print("=> initializing data...")
        datasets, dataloaders = init_data(cfg)
        print("=> start evaluation...")
        trainer.validate(model=model, dataloaders=dataloaders[cfg.data.split], ckpt_path=cfg.model.ckpt_path)
    elif args.task == 'embed_shape':
        print("=> embedding shapes...")
        model.load_from_checkpoint()
        model = model.cuda()
        model.eval()
        model.embed_all_shapes()
    elif args.task == 'retrieve':
        print("=> start inference...")
        model.load_from_checkpoint()
        model = model.cuda()
        model.eval()
        model.retrieve_shapes_from_one_query()