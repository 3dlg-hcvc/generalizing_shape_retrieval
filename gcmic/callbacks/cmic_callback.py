import os
import h5py
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from gcmic.eval.scores import nn_cos_sim
from gcmic.eval.shape_retrieval import evaluate_retrieval_offline, evaluate_retrieval_online


class BaseEvaluateCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        pl_module.mode = "train"
    
    def on_validation_start(self, trainer, pl_module):
        pl_module.mode = "val"
        self.val_batch_output = {"cat_idx": [],
                                 "img_idx": [],
                                 "img_feat": [],
                                 "shape_idx": [],
                                 "shape_feat": [],}
        
    def on_validation_batch_end(self, trainer, pl_module, data_dict, batch, batch_idx, dataloader_idx=0):
        # save image and shape features
        self.val_batch_output["cat_idx"].append(data_dict["cat_idx"])
        self.val_batch_output["img_idx"].append(data_dict["img_idx"])
        self.val_batch_output["img_feat"].append(data_dict["img_feat"])
        self.val_batch_output["shape_idx"].append(data_dict["shape_idx"])
        self.val_batch_output["shape_feat"].append(data_dict["shape_feat"])
        
    def clean_cache(self):
        for k in self.val_batch_output:
            self.val_batch_output[k].clear()
        torch.cuda.empty_cache()


class CMICEvaluate(BaseEvaluateCallback):
        
    def on_validation_epoch_end(self, trainer, pl_module):
        gathered_output = pl_module.all_gather(self.val_batch_output)

        if trainer.is_global_zero:
            all_cat_idx = torch.cat(gathered_output["cat_idx"], dim=0).flatten(0, 1)
            all_img_idx = torch.cat(gathered_output["img_idx"], dim=0).flatten(0, 1)
            all_img_feats = torch.cat(gathered_output["img_feat"], dim=0).flatten(0, 1)
            all_shape_idx = torch.cat(gathered_output["shape_idx"], dim=0).flatten(0, 1)
            all_shape_feats = torch.cat(gathered_output["shape_feat"], dim=0).flatten(0, 1) # (N * mv_num, shape_feat_dim)

            N = len(all_img_feats)
            mv_num = all_shape_feats.shape[0] // N
            all_shape_feats = all_shape_feats.reshape(N, mv_num, -1) # (N, mv_num, shape_feat_dim)

            # there are duplicate shapes to filter out
            unique_shape_idx, shape_idx_start = np.unique(all_shape_idx.detach().cpu().numpy(), return_index=True) # sorted
            shape_idx_start = torch.tensor(shape_idx_start).type_as(all_shape_idx)
            unique_shape_idx = all_shape_idx[shape_idx_start]
            filtered_cat_idx = all_cat_idx[shape_idx_start] # correspond to unique_shape_idx
            filtered_shape_feats = all_shape_feats[shape_idx_start] # (num_unqiue_shape, mv_num, shape_feat_dim)

            num_unique_shapes = len(unique_shape_idx)
            all_sim_scores, pred_shape_idx, pred_cat_idx = [], [], []
            with torch.no_grad():
                cos_sim = nn.CosineSimilarity(dim=-1)
                for i in range(N):
                    img_feat = all_img_feats[i]
                    img_feat_tiled = img_feat.repeat(num_unique_shapes, 1, 1)
                    weighted_shape_feat, _ = pl_module.dot_attention(img_feat_tiled, filtered_shape_feats) # (num_unqiue_shape, 1, shape_feat_dim)
                    sim_scores = cos_sim(img_feat_tiled, weighted_shape_feat).view(-1) # (num_unqiue_shape,)
                    sim_scores_sorted, idx_sorted = torch.sort(sim_scores, descending=True)
                    all_sim_scores.append(sim_scores_sorted.view(1,-1))
                    pred_shape_idx.append(unique_shape_idx[idx_sorted].view(1,-1))
                    pred_cat_idx.append(filtered_cat_idx[idx_sorted].view(1,-1))

            all_sim_scores = torch.cat(all_sim_scores, dim=0)
            pred_shape_idx = torch.cat(pred_shape_idx, dim=0)
            pred_cat_idx = torch.cat(pred_cat_idx, dim=0)

            prediction = {"gt_cat_idx": all_cat_idx,
                        "gt_shape_idx": all_shape_idx,
                        "sim_scores_sorted": all_sim_scores,
                        "pred_shape_idx": pred_shape_idx,
                        "pred_cat_idx": pred_cat_idx}
        
            evaluate_retrieval_online(prediction, pl_module, restrict_cat=True)
        
        self.clean_cache()
        trainer.strategy.barrier()
            

class CMICEvaluateOffline(BaseEvaluateCallback):
    # results could be different (lower) from CMICEvaluate because the latter may not contain all shapes
    
    def on_validation_start(self, trainer, pl_module):
        pl_module.mode = "val"
        self.prediction = {"img_idx": torch.tensor([], dtype=torch.int32, device=pl_module.device),
                            "img_name": [],
                            "gt_cat_idx": torch.tensor([], dtype=torch.int32, device=pl_module.device),
                            "gt_shape_idx": torch.tensor([], dtype=torch.int32, device=pl_module.device),
                            "sim_scores_sorted": torch.tensor([], device=pl_module.device),
                            "pred_shape_idx": torch.tensor([], dtype=torch.int32, device=pl_module.device),
                            "pred_cat_idx": torch.tensor([], dtype=torch.int32, device=pl_module.device),
                            "pose_info": []}

        shape_feats_file = f"shape_feats_{pl_module.cfg.data.shape_feats_source}.h5"
        with h5py.File(os.path.join(os.path.dirname(pl_module.cfg.general.root), f'embed_shape/{shape_feats_file}'), 'r') as f:
            all_shape_ids = [shape_id.decode("utf8") for shape_id in f['shape_ids'][:]]
            self.all_shape_cat_idx = torch.from_numpy(f['cat_idx'][:]).to(pl_module.device)
            self.all_shape_feats = torch.from_numpy(f['shape_feats'][:]).to(pl_module.device)
        if pl_module.cfg.data.extra_shapes:
            for shape_source in pl_module.cfg.data.extra_shapes:
                shape_feats_file = f"shape_feats_{shape_source}.h5"
                with h5py.File(os.path.join(os.path.dirname(pl_module.cfg.general.root), f'embed_shape/{shape_feats_file}'), 'r') as f:
                    all_shape_ids.extend([shape_id.decode("utf8") for shape_id in f['shape_ids'][:]])
                    self.all_shape_cat_idx = torch.cat([self.all_shape_cat_idx, torch.from_numpy(f['cat_idx'][:]).to(pl_module.device)], dim=0)
                    self.all_shape_feats = torch.cat([self.all_shape_feats, torch.from_numpy(f['shape_feats'][:]).to(pl_module.device)], dim=0)
        self.prediction["all_shape_ids"] = all_shape_ids
        
    def on_validation_batch_end(self, trainer, pl_module, data_dict, batch, batch_idx, dataloader_idx=0):
        self.prediction["gt_cat_idx"] = torch.cat([self.prediction["gt_cat_idx"], data_dict["cat_idx"]], dim=0)
        self.prediction["img_idx"] = torch.cat([self.prediction["img_idx"], data_dict["img_idx"]], dim=0)
        self.prediction["img_name"].extend(data_dict["img_name"])
        self.prediction["gt_shape_idx"] = torch.cat([self.prediction["gt_shape_idx"], data_dict["shape_idx"]], dim=0)
        self.prediction["pose_info"].extend(data_dict["pose_info"])
        
        img_feats = data_dict["img_feat"]
        N = len(img_feats)
        
        num_unique_shapes = len(self.all_shape_cat_idx)
        all_sim_scores, pred_shape_idx, pred_cat_idx = [self.prediction["sim_scores_sorted"]], [self.prediction["pred_shape_idx"]], [self.prediction["pred_cat_idx"]]
        with torch.no_grad():
            cos_sim = nn.CosineSimilarity(dim=-1)
            for i in range(N):
                img_feat = img_feats[i]
                img_feat_tiled = img_feat.repeat(num_unique_shapes, 1, 1)
                weighted_shape_feat, _ = pl_module.dot_attention(img_feat_tiled, self.all_shape_feats) # (num_unqiue_shape, 1, shape_feat_dim)
                sim_scores = cos_sim(img_feat_tiled, weighted_shape_feat).view(-1) # (num_unqiue_shape,)
                sim_scores_sorted, idx_sorted = torch.sort(sim_scores, descending=True)
                all_sim_scores.append(sim_scores_sorted.view(1,-1))
                pred_shape_idx.append(idx_sorted.view(1,-1))
                pred_cat_idx.append(self.all_shape_cat_idx[idx_sorted].view(1,-1))
        
        self.prediction["sim_scores_sorted"] = torch.cat(all_sim_scores, dim=0)
        self.prediction["pred_shape_idx"] = torch.cat(pred_shape_idx, dim=0)
        self.prediction["pred_cat_idx"] = torch.cat(pred_cat_idx, dim=0)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        evaluate_retrieval_offline(self.prediction, pl_module, restrict_cat=True, save_metrics=True)