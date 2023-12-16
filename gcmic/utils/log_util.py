import os
import logging
from omegaconf import OmegaConf

from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.loggers.logger import rank_zero_experiment


class TextLogger(object):

    def __init__(self, cfg, ckp_name=None, log_task=None):
        super(TextLogger, self).__init__()
        
        self.cfg = cfg
        self.use_console = cfg.log.use_console_log
        
        self.log_task = log_task if log_task else cfg.general.task
        self.log_name = f'{cfg.general.task}-logger'
        self.ckp_name = ckp_name
        self.log_filenames = {m: f"{self.log_task}.{cfg.data.name}.{'_'.join(cfg.data.annotation_file[:-4].split('_')[2:])}.{cfg.data.test_objects}.{cfg.data.shape_feats_source}.{m}.log" for m in ['full', 'less']}

        self.log_path = cfg.general.root
        self.backup_path = os.path.join(self.log_path, 'backup')
        
        self.logfile_exists = True if self.ckp_name and cfg.general.task == 'train' else False
        
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path, exist_ok=True)
            os.makedirs(self.backup_path, exist_ok=True)

        self._init_logger()

    def _init_logger(self):
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel(logging.DEBUG)

        # self.log_format = '[%(asctime)s  %(levelname)s]\n  %(message)s\n'
        self.log_format = '%(message)s'
        self.formatter = logging.Formatter(self.log_format)
        
        fh_mode = 'w' if self.logfile_exists is None else 'a'
        
        # initialize detailed logging file handler
        self.full_fh = logging.FileHandler(os.path.join(self.log_path, self.log_filenames['full']), mode=fh_mode)
        self.full_fh.setLevel(logging.DEBUG)
        self.full_fh.setFormatter(self.formatter)
        self.logger.addHandler(self.full_fh)
        
        # initialize detailed logging file handler
        self.less_fh = logging.FileHandler(os.path.join(self.log_path, self.log_filenames['less']), mode=fh_mode)
        self.less_fh.setLevel(logging.INFO)
        self.less_fh.setFormatter(self.formatter)
        self.logger.addHandler(self.less_fh)     

        if self.use_console:
            self.ch = logging.StreamHandler()
            self.ch.setLevel(logging.DEBUG)
            self.ch.setFormatter(self.formatter)
            self.logger.addHandler(self.ch)
        
    @classmethod
    def from_checkpoint(cls, cfg):
        ckp_name = cfg.model.use_checkpoint
        logger = cls(cfg, ckp_name)
        return logger
    
    @classmethod
    def from_evaluation(cls, cfg):
        ckp_name = cfg.eval.use_model
        log_task = cfg.eval.task
        logger = cls(cfg, ckp_name, log_task)
        return logger
    
    def store_backup_config(self):
        backup_file = os.path.join(self.backup_path, 'config.yaml')
        OmegaConf.save(self.cfg, backup_file)

    @rank_zero_only
    def info(self, message):
        self.logger.info(message)
    
    @rank_zero_only
    def debug(self, message):
        self.logger.debug(message)
        
    @property
    def name(self):
        return self.log_name

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass