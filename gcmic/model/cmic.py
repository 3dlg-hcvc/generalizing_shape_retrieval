import os
import random
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl

from gcmic.model.layers import DotAttention, MultiHeadAttention, AttentionBlock
from gcmic.model.layers import QueryEncoder, RenderingEncoder, ViTEncoder
from gcmic.loss import CMICLoss, CMICPaperLoss
from gcmic.utils.log_util import TextLogger
from gcmic.utils.img_util import color_tranfer

normal_tf = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


class CMIC(pl.LightningModule):
    """Encoding images and shapes into the same embedding space
        Args:
            cfg: DictConfig
        Return:
            img_feat, shape_feat
    """
    def __init__(self, cfg):
        super(CMIC, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.shape_feature_dim = cfg.model.shape_feature_dim
        self.img_feature_dim = cfg.model.img_feature_dim
        self.lr = self.cfg.train.optim.lr

        if cfg.model.use_vit_backbone:
            self.img_encoder = ViTEncoder(in_dim=cfg.model.img_in_dim)
            self.shape_encoder = RenderingEncoder(in_dim=cfg.model.shape_in_dim)
        elif cfg.model.use_r18_backbone:
            self.img_encoder = QueryEncoder(in_dim=cfg.model.img_in_dim, backbone="resnet18")
            self.shape_encoder = RenderingEncoder(in_dim=cfg.model.shape_in_dim)
        else:
            self.img_encoder = QueryEncoder(in_dim=cfg.model.img_in_dim)
            self.shape_encoder = RenderingEncoder(in_dim=cfg.model.shape_in_dim)
        if cfg.model.use_attn_block:
            self.dot_attention = AttentionBlock(self.img_feature_dim)
        elif cfg.model.use_multihead_attn:
            self.dot_attention = MultiHeadAttention(self.img_feature_dim)
        else:
            self.dot_attention = DotAttention(self.img_feature_dim)
        
        # self._init_random_seed()
        self._init_criterion()
        if cfg.general.task not in ["train", "finetune"]:
            self._init_text_logger()
        if cfg.general.task == "train" and cfg.model.use_paper_loss:
            self._init_shape_loader()
    
    
    def _init_random_seed(self):
        print("=> setting random seed...")
        if self.cfg.general.manual_seed:
            random.seed(self.cfg.general.manual_seed)
            np.random.seed(self.cfg.general.manual_seed)
            torch.manual_seed(self.cfg.general.manual_seed)
            torch.cuda.manual_seed_all(self.cfg.general.manual_seed)
            
    
    def _init_text_logger(self):
        self.text_logger = TextLogger(self.cfg, self.cfg.model.use_checkpoint)
        
    
    def _init_shape_loader(self):
        if self.cfg.data.name == 'pix3d':
            from gcmic.dataset.pix3d_shape import Pix3DShape as ShapeDataset
        elif self.cfg.data.name == 'scan2cad':
            from gcmic.dataset.scan2cad_shape import Scan2CADShape as ShapeDataset
        elif self.cfg.data.name == 'moos':
            from gcmic.dataset.future3d import Future3D as ShapeDataset
        self.shape_dataset = ShapeDataset(self.cfg)
        self.shape_loader = DataLoader(self.shape_dataset, batch_size=self.cfg.data.batch_size, shuffle=True, drop_last=False)
        self.iter_shape = iter(self.shape_loader)
        
        
    def _init_criterion(self):
        if self.cfg.model.use_paper_loss:
            self.criterion = CMICPaperLoss(self.cfg.model.beta, self.cfg.model.tau)
        else:
            self.criterion = CMICLoss(self.cfg.model.beta, self.cfg.model.tau)


    def configure_optimizers(self):
        print("=> configure optimizer...")
        optim_class_name = self.cfg.train.optim.classname
        optim = getattr(torch.optim, optim_class_name)
        if optim_class_name == "Adam":
            optimizer = optim(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.5, 0.999))
        elif optim_class_name == "SGD":
            optimizer = optim(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, momentum=self.cfg.train.optim.momentum, weight_decay=self.cfg.train.optim.weight_decay)
        else:
            raise NotImplemented
        return {"optimizer": optimizer}
    
    
    def _loss(self, data_dict):
        data_dict['loss'] = self.criterion(data_dict, self.mode)
        return data_dict
    
    
    def prepare_img_input(self, data_dict):
        data_dict["img_input"] = torch.cat([data_dict["image"], data_dict["mask"]], dim=1).type_as(data_dict["image"]) # (B, 4, 224, 224)
        
        
    def apply_color_transfer(self, data_dict):
        seq = torch.randperm(data_dict["image"].shape[0])
        style_img = data_dict["image"][seq]
        transfer_img = color_tranfer(style_img.cpu(), data_dict["image"].cpu())
        transfer_img = normal_tf(transfer_img).detach()
        data_dict["style_img"] = data_dict["image"][seq]
        data_dict["tgt_img"] = data_dict["image"]
        data_dict["image"] = transfer_img.to(data_dict["tgt_img"])
        
        
    def forward(self, data_dict):
        batch_size = data_dict["data_idx"].shape[0]
        
        img_feat = self.img_encoder(data_dict["img_input"]) # (B, img_feat_dim)
        shape_feat = self.shape_encoder(data_dict["shape"]) # (B * mv_num, shape_feat_dim)
        
        img_feat_tiled = img_feat.unsqueeze(0).repeat(batch_size, 1, 1) # (B, B, img_feat_dim)
        shape_feat_expanded = shape_feat.view(batch_size, -1, self.shape_feature_dim) # (B, mv_num, shape_feat_dim)
        weighted_shape_feat, _ = self.dot_attention(img_feat_tiled, shape_feat_expanded) # (batch_size, num_img, shape_feat_dim)

        data_dict["img_feat"] = img_feat
        data_dict["shape_feat"] = shape_feat
        data_dict["shape_context_feats"] = weighted_shape_feat # (B, B, shape_feat_dim)
        
        return data_dict
    
    def prepare_unique_shape(self, data_dict):
        bs = data_dict["data_idx"].shape[0]
        # get Instance and Category level label
        inst_list = []  # record the unique idx for each model
        inst_index = [] # instance level idx
        idx_list = []   # using for torch.cat
        
        for ii in range(bs):
            tmp_cat = data_dict['cat_idx'][ii].item()
            tmp_inst = data_dict['shape_idx'][ii].item()
            try:
                # model already existed
                idx = inst_list.index((tmp_cat, tmp_inst))
                inst_index.append(idx)
            except ValueError: 
                inst_index.append(len(inst_list))
                inst_list.append((tmp_cat, tmp_inst))
                idx_list.append(ii)
        rendering_img = torch.cat([data_dict['shape'][idx:idx+1] for idx in idx_list], dim=0)
        
        while not len(inst_list) == bs:
            try:
                shape_meta = next(self.iter_shape)
            except StopIteration:
                self.iter_shape = iter(self.shape_loader)
                shape_meta = next(self.iter_shape)
            
            tmp_cats = shape_meta['cat_idx']
            tmp_insts = shape_meta['shape_idx']
            tmp_reinderings = shape_meta['shape']
            tmp_rendering_list = []

            for ii in range(len(tmp_cats)):
                tmp_cat = tmp_cats[ii].item()
                tmp_inst = tmp_insts[ii].item()
                try:
                    # model already existed
                    idx = inst_list.index((tmp_cat, tmp_inst))
                except ValueError:
                    inst_list.append((tmp_cat, tmp_inst))
                    tmp_rendering_list.append(tmp_reinderings[ii:ii+1]) 
                if len(inst_list) == bs:
                    break
            if not len(tmp_rendering_list) == 0:
                tmp_reindering = torch.cat(tmp_rendering_list, dim=0).to(data_dict['shape'])
                rendering_img = torch.cat([rendering_img, tmp_reindering], dim=0)
        data_dict['shape'] = rendering_img
        
        cats_list = []
        shape_cats_index = []
        image_cats_index = []
        for ii, items in enumerate(inst_list):
            tmp_cat, tmp_inst = items
            try:
                idx = cats_list.index(tmp_cat)
                shape_cats_index.append(idx)
            except ValueError:
                shape_cats_index.append(len(cats_list))
                cats_list.append(tmp_cat)
            tmp_cat, tmp_inst = inst_list[inst_index[ii]]
            idx = cats_list.index(tmp_cat)
            image_cats_index.append(idx)

        inst_label = torch.tensor(inst_index, dtype=torch.long).view(-1,1)
        InstsMat = torch.zeros((bs, bs)).scatter_(1, inst_label, torch.ones((bs, bs))).transpose(0, 1)

        shape_cats_label = torch.tensor(shape_cats_index, dtype=torch.int)
        image_cats_label = torch.tensor(image_cats_index, dtype=torch.int)

        shape_cats_labels = shape_cats_label.unsqueeze(1).repeat(1, bs)
        image_cats_labels = image_cats_label.unsqueeze(0).repeat(bs, 1)
        CatsMat = torch.tensor(shape_cats_labels==image_cats_labels, dtype=torch.float)

        # ######## [ end ] ######## [create no repeat rendering batch data ] ########
        # InstsMat      bs, bs  (shape, image)
        # CatsMat       bs, bs  (shape, image)
        data_dict['InstsMat'] = InstsMat.to(data_dict['shape'])
        data_dict['CatsMat'] = CatsMat.to(data_dict['shape'])
    
    
    def training_step(self, data_dict, idx):
        if self.cfg.model.use_paper_loss:
            self.prepare_unique_shape(data_dict)
        if self.cfg.data.use_color_transfer:
            self.apply_color_transfer(data_dict)
        self.prepare_img_input(data_dict)
        
        mv = data_dict["shape"]
        data_dict["shape"] = mv.reshape(-1, mv.shape[2], mv.shape[3], mv.shape[4])
        
        data_dict = self.forward(data_dict)
        data_dict = self._loss(data_dict)
        loss = data_dict["loss"]

        in_prog_bar = ["loss", "inst_loss", "cat_loss", "shape_loss"]
        for key, value in data_dict.items():
            if "loss" in key:
                self.log("train/{}".format(key), value.item(), prog_bar=key in in_prog_bar, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    
    def validation_step(self, data_dict, idx):
        self.prepare_img_input(data_dict)
        
        mv = data_dict["shape"]
        data_dict["shape"] = mv.reshape(-1, mv.shape[2], mv.shape[3], mv.shape[4])
        
        data_dict = self.forward(data_dict)
        # data_dict = self._loss(data_dict)

        # in_prog_bar = ["total_loss"]
        # for key, value in data_dict.items():
        #     if "loss" in key:
        #         self.log("val/{}".format(key), value.item(), prog_bar=key in in_prog_bar, on_step=False, on_epoch=True, sync_dist=True)
        
        return data_dict
    
    
    def embed_all_shapes(self):
        if self.cfg.data.name == 'pix3d':
            from gcmic.dataset.pix3d_shape import Pix3DShape as ShapeDataset
            shape_feats_file = "shape_feats_pix3d.h5"
        elif self.cfg.data.name == 'scan2cad':
            from gcmic.dataset.scan2cad_shape import Scan2CADShape as ShapeDataset
            shape_feats_file = "shape_feats_scan2cad.h5"
        elif self.cfg.data.name == 'moos':
            from gcmic.dataset.future3d import Future3D as ShapeDataset
            shape_feats_file = "shape_feats_moos.h5"
        
        shape_dataset = ShapeDataset(self.cfg)
        shape_loader = DataLoader(shape_dataset, batch_size=self.cfg.data.batch_size, shuffle=False, drop_last=False)

        with torch.no_grad():
            with h5py.File(os.path.join(self.cfg.general.root, shape_feats_file), 'w') as f:
                f.create_dataset(
                    "cat_idx",
                    shape=(len(shape_dataset),),
                    dtype=np.int32,
                    compression='gzip',
                    compression_opts=9,
                )
                f.create_dataset(
                    "shape_ids",
                    shape=(len(shape_dataset),),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                )
                f.create_dataset(
                    "shape_feats",
                    shape=(len(shape_dataset), 12, 128),
                    dtype=np.float32,
                    compression='gzip',
                    compression_opts=9,
                )
                for batch in tqdm(shape_loader):
                    cat_idx = batch["cat_idx"]
                    idx = batch["shape_idx"].cpu().numpy()
                    f["cat_idx"][idx] = cat_idx.cpu().numpy()
                    f["shape_ids"][idx] = batch["shape_id"]
                    
                    mv = batch["shape"]
                    batch["shape"] = mv.reshape(-1, mv.shape[2], mv.shape[3], mv.shape[4]).to(self.device)
                    shape_feat = self.shape_encoder(batch["shape"]) # (B * mv_num, shape_feat_dim)
                    shape_feat = shape_feat.reshape(-1, mv.shape[1], 128)
                    f["shape_feats"][idx] = shape_feat.cpu().numpy()
                    
                    
    def load_from_checkpoint(self):
        assert self.cfg.model.ckpt_path is not None
        self.text_logger.info(f"=> loading shape retrieval checkpoint from {self.cfg.model.ckpt_path} ...")
        checkpoint = torch.load(self.cfg.model.ckpt_path)
        sync_state_dict = {k.replace('model.', ''): v for k, v in checkpoint["state_dict"].items()}
        self.load_state_dict(sync_state_dict)
        
    
    @torch.no_grad()
    def retrieve_shapes_from_one_query(self, img_path, mask_path, shape_feats_file="shape_feats_moos.h5"):
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        mask = Image.open(mask_path).convert("L").resize((224, 224), Image.NEAREST)
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img).unsqueeze(0).to(self.device)
        mask = transform(mask).unsqueeze(0).to(self.device)
        
        img_input = torch.cat([img, mask], dim=1).type_as(img)
        img_feat = self.img_encoder(img_input) # (B, img_feat_dim)
        
        with h5py.File(os.path.join(os.path.dirname(self.cfg.general.root), f'embed_shape/{shape_feats_file}.h5'), 'r') as f:
            all_shape_cat_idx = torch.from_numpy(f['cat_idx'][:]).to(self.device)
            all_shape_ids = [name.decode("utf8") for name in f['shape_ids'][:]]
            all_shape_feats = torch.from_numpy(f['shape_feats'][:]).to(self.device)
        
        num_unique_shapes = len(all_shape_cat_idx)
        img_feat_tiled = img_feat.repeat(num_unique_shapes, 1, 1)
        weighted_shape_feat, _ = self.dot_attention(img_feat_tiled, all_shape_feats) # (num_unqiue_shape, 1, shape_feat_dim)
        cos_sim = nn.CosineSimilarity(dim=-1)
        sim_scores = cos_sim(img_feat_tiled, weighted_shape_feat).view(-1) # (num_unqiue_shape,)
        _, pred_shape_idx = torch.sort(sim_scores, descending=True)
        
        retr_objs = [all_shape_ids[int(obj_idx)] for obj_idx in pred_shape_idx]
        print(retr_objs[:5])