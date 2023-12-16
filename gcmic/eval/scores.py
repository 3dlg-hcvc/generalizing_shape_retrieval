import logging
import numpy as np
from collections import defaultdict

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.ops import knn_gather, knn_points
from pytorch3d.structures import Meshes

logger = logging.getLogger(__name__)


def exp_cos_sim(img_feature, shape_features, temperature=0.15):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    img_feature_repeated = img_feature.repeat(shape_features.size()[0], 1)
    assert list(img_feature_repeated.size()) == [shape_features.size()[0], 128], 'incorrect image feature shape'
    
    return torch.exp(cos(img_feature_repeated, shape_features) * (1 / temperature))


def nn_cos_sim(feats1, feats2, use_exp=True, temperature=0.15, by_chunks=False):
    """
    Args:
        feats1: (N,C) torch tensor
        feats2: (M,C) torch tensor
        use_exp: bool, whether to use exponential cos sim
        temperature: scalar, the temperature used in exponential cos sim
    return:
        sim_mat: (N,M) torch float32, similarity matrix
        idx1: (N,) torch int64 tensor
    """
    cos_sim = nn.CosineSimilarity(dim=-1)
    if not by_chunks:
        N = feats1.shape[0]
        M = feats2.shape[0]
        feats1_expand_tile = feats1.unsqueeze(1).repeat(1,M,1) # (N, M, C)
        feats2_expand_tile = feats2.unsqueeze(0).repeat(N,1,1) # (N, M, C)
        sim_mat = cos_sim(feats1_expand_tile, feats2_expand_tile) # (N, M)
    else:
        sim_mat_chunks = []
        N = feats1.shape[0]
        num_chunks = by_chunks
        chunk_size = feats2.shape[0] // num_chunks
        feats1_expand_tile = feats1.unsqueeze(1).repeat(1,chunk_size,1) # (N, chunk_size, C)
        for c in range(num_chunks):
            feats2_chunk = feats2[chunk_size*c:chunk_size*(c+1)] # (chunk_size, C)
            feats2_expand_tile = feats2_chunk.unsqueeze(0).repeat(N,1,1) # (N, chunk_size, C)
            sim_mat_chunks.append(cos_sim(feats1_expand_tile, feats2_expand_tile)) # append (N, chunk_size)
        sim_mat = torch.cat(sim_mat_chunks, dim=1)
    if use_exp:
        return torch.exp(sim_mat * (1 / temperature))
    else:
        return sim_mat


def compute_shape_scores(gt_mesh, pred_meshes, device, thresholds=[0.1, 0.3, 0.5]):
    gt_verts, gt_faces = (
            gt_mesh[0].clone(),
            Variable(torch.rand(1, 3)).clone(),
        )
    gt_verts = gt_verts.to(device)
    gt_faces = gt_faces.to(device)
    gt_mesh_inst = Meshes(verts=[gt_verts], faces=[gt_faces]).to(device)
    pred_verts = [mesh[0].to(device) for mesh in pred_meshes]
    pred_faces = [Variable(torch.rand(1, 3)).to(device) for _ in pred_meshes]
    pred_mesh_insts = Meshes(verts=pred_verts, faces=pred_faces).to(device)
    
    gt_normals = gt_mesh[1].to(device)
    pred_normals = torch.cat([mesh[1].unsqueeze(0) for mesh in pred_meshes], 0).to(device)
    metrics = compare_meshes(pred_mesh_insts, pred_normals, gt_mesh_inst, gt_normals, thresholds=thresholds, reduce=False)   
    
    return metrics


@torch.no_grad()
def compare_meshes(
    pred_meshes, pred_normals, gt_meshes, gt_normals, num_samples=10000, scale="gt-10", thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], reduce=True, eps=1e-8
):
    """
    Compute evaluation metrics to compare meshes. We currently compute the
    following metrics:
    - L2 Chamfer distance
    - Normal consistency
    - Absolute normal consistency
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    Inputs:
        - pred_meshes (Meshes): Contains N predicted meshes
        - gt_meshes (Meshes): Contains 1 or N ground-truth meshes. If gt_meshes
          contains 1 mesh, it is replicated N times.
        - num_samples: The number of samples to take on the surface of each mesh.
          This can be one of the following:
            - (int): Take that many uniform samples from the surface of the mesh
            - 'verts': Use the vertex positions as samples for each mesh
            - A tuple of length 2: To use different sampling strategies for the
              predicted and ground-truth meshes (respectively).
        - scale: How to scale the predicted and ground-truth meshes before comparing.
          This can be one of the following:
            - (float): Multiply the vertex positions of both meshes by this value
            - A tuple of two floats: Multiply the vertex positions of the predicted
              and ground-truth meshes by these two different values
            - A string of the form 'gt-[SCALE]', where [SCALE] is a float literal.
              In this case, each (predicted, ground-truth) pair is scaled differently,
              so that bounding box of the (rescaled) ground-truth mesh has longest
              edge length [SCALE].
        - thresholds: The distance thresholds to use when computing precision, recall,
          and F1 scores.
        - reduce: If True, then return the average of each metric over the batch;
          otherwise return the value of each metric between each predicted and
          ground-truth mesh.
        - eps: Small constant for numeric stability when computing F1 scores.
    Returns:
        - metrics: A dictionary mapping metric names to their values. If reduce is
          True then the values are the average value of the metric over the batch;
          otherwise the values are Tensors of shape (N,).
    """
    pred_meshes, gt_meshes = _scale_meshes(pred_meshes, gt_meshes, scale)
    pred_points = torch.cat([verts.unsqueeze(0) for verts in pred_meshes.verts_list()], 0)
    gt_points = gt_meshes.verts_list()[0]

    # Will add a dimension at axis=0 for gt_mesh!
    if len(gt_meshes) == 1:
        # (1, S, 3) to (N, S, 3)
        gt_points = gt_points.expand(len(pred_meshes), -1, -1)
        gt_normals = gt_normals.expand(len(pred_meshes), -1, -1)

    if torch.is_tensor(pred_points) and torch.is_tensor(gt_points):
        # We can compute all metrics at once in this case
        metrics = _compute_sampling_metrics(
            pred_points, pred_normals, gt_points, gt_normals, thresholds, eps
        )
    else:
        # Slow path when taking vert samples from non-equisized meshes; we need to iterate over the batch
        metrics = defaultdict(list)
        for cur_points_pred, cur_points_gt in zip(pred_points, gt_points):
            cur_metrics = _compute_sampling_metrics(
                cur_points_pred[None], None, cur_points_gt[None], None, thresholds, eps
            )
            for k, v in cur_metrics.items():
                metrics[k].append(v.item())
        metrics = {k: torch.tensor(vs) for k, vs in metrics.items()}

    if reduce:
        # Average each metric over the batch
        metrics = {k: v.mean().item() for k, v in metrics.items()}

    return metrics


def _scale_meshes(pred_meshes, gt_meshes, scale):
    if isinstance(scale, float):
        # Assume scale is a single scalar to use for both preds and GT
        pred_scale = gt_scale = scale
    elif isinstance(scale, tuple):
        # Rescale preds and GT with different scalars
        pred_scale, gt_scale = scale
    elif scale.startswith("gt-"):
        # Rescale both preds and GT so that the largest edge length of each GT mesh is target
        target = float(scale[3:])
        bbox = gt_meshes.get_bounding_boxes()  # (N, 3, 2)
        long_edge = (bbox[:, :, 1] - bbox[:, :, 0]).max(dim=1)[0]  # (N,)
        scale = target / long_edge
        if scale.numel() == 1:
            scale = scale.expand(len(pred_meshes))
        pred_scale, gt_scale = scale, scale
    else:
        raise ValueError("Invalid scale: %r" % scale)
    pred_meshes = pred_meshes.scale_verts(pred_scale)
    gt_meshes = gt_meshes.scale_verts(gt_scale)
    return pred_meshes, gt_meshes


def _compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, thresholds, eps):
    """
    Compute metrics that are based on sampling points and normals:
    - L2 Chamfer distance
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    - Normal consistency (if normals are provided)
    - Absolute normal consistency (if normals are provided)
    Inputs:
        - pred_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each predicted mesh
        - pred_normals: Tensor of shape (N, S, 3) giving normals of points sampled
          from the predicted mesh, or None if such normals are not available
        - gt_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each ground-truth mesh
        - gt_normals: Tensor of shape (N, S, 3) giving normals of points sampled from
          the ground-truth verts, or None of such normals are not available
        - thresholds: Distance thresholds to use for precision / recall / F1
        - eps: epsilon value to handle numerically unstable F1 computation
    Returns:
        - metrics: A dictionary where keys are metric names and values are Tensors of
          shape (N,) giving the value of the metric for the batch
    """
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )
    
    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)
    if gt_normals is not None:
        pred_normals_near = knn_gather(gt_normals, knn_pred.idx, lengths_gt)[..., 0, :]  # (N, S, 3)
    else:
        pred_normals_near = None

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)
    if pred_normals is not None:
        gt_normals_near = knn_gather(pred_normals, knn_gt.idx, lengths_pred)[..., 0, :]  # (N, S, 3)
    else:
        gt_normals_near = None

    # Compute L2 chamfer distances
    chamfer_l2 = pred_to_gt_dists2.mean(dim=1) + gt_to_pred_dists2.mean(dim=1)
    metrics["CD"] = chamfer_l2

    # Compute normal consistency and absolute normal consistance only if
    # we actually got normals for both meshes
    if pred_normals is not None and gt_normals is not None:
        pred_to_gt_cos = F.cosine_similarity(pred_normals, pred_normals_near, dim=2)
        gt_to_pred_cos = F.cosine_similarity(gt_normals, gt_normals_near, dim=2)
        
        pred_to_gt_cos_sim = pred_to_gt_cos.mean(dim=1)
        pred_to_gt_abs_cos_sim = pred_to_gt_cos.abs().mean(dim=1)
        gt_to_pred_cos_sim = gt_to_pred_cos.mean(dim=1)
        gt_to_pred_abs_cos_sim = gt_to_pred_cos.abs().mean(dim=1)
        normal_dist = 0.5 * (pred_to_gt_cos_sim + gt_to_pred_cos_sim)
        abs_normal_dist = 0.5 * (pred_to_gt_abs_cos_sim + gt_to_pred_abs_cos_sim)
        metrics["NC"] = normal_dist
        metrics["AbsNC"] = abs_normal_dist

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics[f"Precision@{t}"] = precision
        metrics[f"Recall@{t}"] = recall
        metrics[f"F1@{t}"] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics


def compute_ap(scores, labels, npos, device=None):
    if device is None:
        device = scores.device

    if len(scores) == 0:
        return 0.0
    tp = labels == 1
    fp = labels == 0
    sc = scores
    assert tp.size() == sc.size()
    assert tp.size() == fp.size()
    sc, ind = torch.sort(sc, descending=True)
    tp = tp[ind].to(dtype=torch.float32)
    fp = fp[ind].to(dtype=torch.float32)
    tp = torch.cumsum(tp, dim=0)
    fp = torch.cumsum(fp, dim=0)

    # # Compute precision/recall
    rec = tp / npos
    prec = tp / (fp + tp)
    ap = xVOCap(rec, prec, device)

    return ap


# https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
# PASCAL VOC
def xVOCap(rec, prec, device):
    z = rec.new_zeros((1))
    o = rec.new_ones((1))
    mrec = torch.cat((z, rec, o))
    mpre = torch.cat((z, prec, z))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    I = (mrec[1:] != mrec[0:-1]).nonzero(as_tuple=False)[:, 0] + 1
    ap = 0
    for i in I:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap
