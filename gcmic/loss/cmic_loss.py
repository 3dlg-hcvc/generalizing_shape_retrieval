import torch
import torch.nn as nn

    
class CMICPaperLoss(nn.Module):
    """Cross-Modal Instance and Category Contrastive Loss
        Hyperparameters are from the paper 
    """
    def __init__(self, beta=0.2, tau=0.1):
        super(CMICPaperLoss, self).__init__()
        self.beta = beta
        self.tau = tau
        
    def forward(self, data_dict, mode):
        img_features = data_dict["img_feat"] # (B, img_feat_dim), query_feats, -> v^q_i
        shape_features = data_dict["shape_context_feats"] # (B, B, shape_feat_dim), context_feats, -> v^r_ij
        
        N = len(img_features)
        img_features = img_features.unsqueeze(0).repeat(N, 1, 1)
        
        sims = (img_features * shape_features).sum(dim=2)  # (N, N) 
        # sims = self.cos_sim(img_features.unsqueeze(0), shape_features) # v^q_i . v^r_ij
        sims_exp = torch.exp(sims / self.tau) # (N, N) -> (shape, image)
        
        InstsMat = data_dict['InstsMat']
        instance_loss = -torch.log((sims_exp * InstsMat).sum(dim=0) / sims_exp.sum(dim=0)).mean()
        data_dict["inst_loss"] = instance_loss
        
        if len(torch.unique(data_dict["cat_idx"])) == 1:
            data_dict["cat_loss"] = torch.tensor(0.)
            return instance_loss
        
        CatsMat = data_dict['CatsMat']
        pos_num = CatsMat.sum(dim=0)
        pos_num[pos_num==0] = 1 # In some cases, pos_num = 0  -->> nan
        SumMat = sims_exp.sum(dim=0).view(1, -1).repeat(N, 1) # SumMat excluding InstMat
        ExcProdMat = sims_exp * CatsMat / SumMat

        ExcProdMat[ExcProdMat==0] = 1
        # ExcProdMat[ExcProdMat<1e-5] = 1e-5
        loss_cats_ = -torch.log(ExcProdMat).sum(dim=0)/pos_num
        loss_cats = loss_cats_[loss_cats_ != 0]
        category_loss = loss_cats.mean()
        data_dict["cat_loss"] = category_loss
        
        # CMIC loss
        loss = instance_loss + self.beta * category_loss
            
        return loss


class CMICLoss(nn.Module):
    """Cross-Modal Instance and Category Contrastive Loss
        Hyperparameters are from the paper 
    """
    def __init__(self, beta1=0.2, tau=0.1, beta2=0.2, filter_repeated_pos=False):
        super(CMICLoss, self).__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.filter_repeated_pos = filter_repeated_pos
        
    def forward(self, data_dict, mode):
        img_features = data_dict["img_feat"] # (B, img_feat_dim), query_feats, -> v^q_i
        shape_features = data_dict["shape_context_feats"] # (B, B, shape_feat_dim), context_feats, -> v^r_ij
        
        N = len(img_features)
        img_features = img_features.unsqueeze(0).repeat(N, 1, 1)
        
        sims = (img_features * shape_features).sum(dim=2)  # (N, N) -> (shape, image)
        # sims = self.cos_sim(img_features.unsqueeze(0), shape_features) # v^q_i . v^r_ij
        sims_exp = torch.exp(sims / self.tau) # (shape, image)
        
        # Instance Loss
        # pos_mask = torch.eye(N, dtype=torch.bool).type_as(sims) # (N, N)
        # if mode != "train" or self.filter_repeated_pos:
        gt_shape_idx = data_dict["shape_idx"] # (M,)
        gt_shape_idx_tile = gt_shape_idx.unsqueeze(0).repeat((N, 1)) # (N, M)
        pos_mask = (gt_shape_idx_tile == gt_shape_idx.view(-1, 1)).type_as(sims) # (N, N), -> (shape, image)
        instance_loss = - torch.log((pos_mask*sims_exp).sum(dim=0) / sims_exp.sum(dim=0)).mean()
        data_dict["inst_loss"] = instance_loss
        
        if len(torch.unique(data_dict["cat_idx"])) == 1:
            data_dict["cat_loss"] = torch.tensor(0.)
            return instance_loss

        # Category Loss
        gt_cat_idx = data_dict["cat_idx"]
        gt_cat_idx_tile = gt_cat_idx.unsqueeze(0).repeat((N, 1)) # (N, M)
        cat_mask = (gt_cat_idx_tile == gt_cat_idx.view(-1, 1)).type_as(sims) # (N, N) -> (shape, image)
        cat_mask = cat_mask.fill_diagonal_(False)
        cat_num = cat_mask.sum(0)
        cat_num = cat_num.masked_fill_(cat_num==0, 1)
        
        soft_prob = cat_mask * sims_exp / sims_exp.sum(0)
        soft_prob = soft_prob.masked_fill_(~cat_mask.bool(), 1)
        category_loss = -torch.log(soft_prob).sum(0) / cat_num
        category_loss = category_loss[category_loss != 0].mean()
        data_dict["cat_loss"] = category_loss
        
        # CMIC loss
        loss = instance_loss + self.beta1 * category_loss
            
        return loss