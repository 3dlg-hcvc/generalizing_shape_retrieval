import os, json
import h5py
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

import torchvision.transforms.functional as TF
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gcmic.eval.util import prepare_pix3d_gt, prepare_scan2cad_gt, prepare_moos_gt
from gcmic.eval.scores import compute_shape_scores, compute_ap
from gcmic.utils.pt3d_util import render_obj_in_pix3d_world, render_obj_in_scan2cad_world, render_obj_in_moos_world
from gcmic.utils.lfd_util import extract_efd_feature, extract_zernike_moments_feature
from gcmic.utils.img_util import resize_padding, get_largest_contour

CLASS_TO_IDX = {"chair": 0, "bed": 1, "sofa": 2,"table": 3}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


def save_all_metrics(metrics, predictions, pl_module, top_k=5, restrict_cat=True):
    cfg = pl_module.cfg
    filename = f"metrics.{cfg.data.name}.{'_'.join(cfg.data.annotation_file[:-4].split('_')[2:])}.{cfg.data.test_objects}.{cfg.data.shape_feats_source}.json"
    file_path = os.path.join(cfg.general.root, filename)

    metric_names = ['CD', 'LFD', 'MIoU', 'vLFD', 'LPIPS']
    metric_dict = {}
    for key in metric_names:
        if key in metrics:
            metric_dict[key] = metrics[key].tolist()
    
    if restrict_cat:   
        metric_dict['retrieval'] = []
        metric_dict['view_idx'] = []
        for idx, cat_idx_mask in enumerate(predictions["cat_idx_mask"]):
            metric_dict['retrieval'].append(predictions['pred_shape_idx'][idx, cat_idx_mask[:top_k].view(-1)].tolist())
            if "pred_view_idx" in predictions:
                metric_dict['view_idx'].append(predictions['pred_view_idx'][idx, cat_idx_mask[:top_k].view(-1)].tolist())
    else:
        metric_dict['retrieval'] = predictions['pred_shape_idx'][:, :top_k].tolist()
        if "pred_view_idx" in predictions:
            metric_dict['view_idx'] = predictions['pred_view_idx'][:, :top_k].tolist()
    
    with open(file_path, "w") as f:
        json.dump(metric_dict, f)


def evaluate_render_view(predictions, best_shape_idx_topk, cfg, max_size=1024, device='cpu'):
    out_dir = f"{cfg.general.root}/examples"
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    render_bs = 5
    mask_ious, lfd_l1s, lpips_scores = [], [], []
    with torch.no_grad():
        for i in tqdm(range(len(predictions["img_idx"]))):
            torch.cuda.empty_cache()
            pose_info = predictions["pose_info"][i]
            img_size = tuple(pose_info["img_size"])
            if max(img_size) > max_size:
                ratio = float(max_size) / max(img_size)
                img_size = tuple(int(x * ratio) for x in img_size)
                pose_info["img_size"] = img_size
            
            if cfg.data.name == 'pix3d':
                img_name = predictions["img_name"][i]
                gt_rgb_crop, gt_mask_t, occlusion_mask, bbox, gt_mask_lfd = prepare_pix3d_gt(img_name, pose_info, device, out_dir, cfg.eval.save_vis)
                render_obj = render_obj_in_pix3d_world
            elif cfg.data.name == 'scan2cad':
                gt_rgb_crop, gt_mask_t, occlusion_mask, bbox, gt_mask_lfd = prepare_scan2cad_gt(pose_info, device, out_dir, cfg.eval.save_vis)
                render_obj = render_obj_in_scan2cad_world
            elif cfg.data.name == 'moos':
                gt_rgb_crop, gt_mask_t, occlusion_mask, bbox, gt_mask_lfd = prepare_moos_gt(pose_info, device, out_dir, cfg.eval.save_vis)
                render_obj = render_obj_in_moos_world
            
            x, y, w, h = bbox
            mask_ious_topk, lfd_l1s_topk, lpips_scores_topk = [], [], []
            for j in range(0, len(best_shape_idx_topk[i]), render_bs):
                obj_indices = best_shape_idx_topk[i, j:j+render_bs].cpu().tolist()
                obj_names = [predictions["all_shape_ids"][obj_idx] for obj_idx in obj_indices]
                obj_names.append(predictions["all_shape_ids"][predictions['gt_shape_idx'][i].item()])
                images = render_obj(obj_names, pose_info, device=device)
                rgbs_np = (images["rgb"][..., :3].cpu().numpy() * 255).astype(np.uint8)
                masks_np = images["mask"].cpu().numpy()
                
                mask_iou = (torch.logical_and(images["mask"], gt_mask_t.unsqueeze(0)).sum([1,2]) / (torch.logical_or(images["mask"], gt_mask_t.unsqueeze(0)).sum([1,2]) + 1e-6)) # some masks are empty because objects are not captured
                mask_ious_topk.extend(mask_iou.cpu().tolist())
                
                rgbs_mask = occlusion_mask[None, ...] & masks_np.astype(bool)
                rgbs_np = rgbs_np * rgbs_mask[..., None]
                rgbs_np[~rgbs_mask] = [255, 255, 255]
                for k in range(render_bs):
                    rgb_crop = rgbs_np[k, y:y + h, x:x + w, :]
                    rgb_crop_im = resize_padding(Image.fromarray(rgb_crop), 100, mode="RGB", background_color=(255, 255, 255))
                    rgb_crop = TF.to_tensor(rgb_crop_im).to(device)
                    lpips_score = lpips(gt_rgb_crop.unsqueeze(0), rgb_crop.unsqueeze(0))
                    lpips_scores_topk.append(lpips_score.item())

                    mask_np = masks_np[k].astype(np.uint8)
                    # if gt_feat is not None and mask_np.any():
                    if gt_mask_lfd is not None and mask_np.sum() > 20:
                        render_fourier = extract_efd_feature(get_largest_contour(mask_np))
                        render_moments = extract_zernike_moments_feature(mask_np, radius=128, degree=10)
                        render_lfd = np.concatenate([render_fourier, render_moments])
                        lfd_l1s_topk.append((np.abs(gt_mask_lfd-render_lfd).sum()))
                    else:
                        lfd_l1s_topk.append(5) # some masks are empty because objects are not captured
                
            mask_ious.append(mask_ious_topk)
            lfd_l1s.append(lfd_l1s_topk)
            lpips_scores.append(lpips_scores_topk)
        
    mask_ious = torch.tensor(mask_ious)
    lfd_l1s = torch.tensor(lfd_l1s)
    lpips_scores = torch.tensor(lpips_scores)
    return mask_ious, lfd_l1s, lpips_scores
    

def evaluate_retrieval_online(predictions, pl_module, top_k=5, restrict_cat=False):
    cfg = pl_module.cfg
    
    gt_shape_idx = predictions['gt_shape_idx']
    gt_cat_idx = predictions['gt_cat_idx']
    best_cat_idx = predictions['pred_cat_idx'][:, 0]
    if restrict_cat:
        best_shape_idx_topk = torch.zeros_like(predictions['pred_shape_idx'][:, :top_k])
        best_shape_idx = torch.zeros_like(predictions['pred_shape_idx'][:, 0])
        for idx, cat_idx in enumerate(predictions["gt_cat_idx"]):
            cat_idx_mask = torch.nonzero(predictions['pred_cat_idx'][idx] == cat_idx)
            best_shape_idx_topk[idx] = predictions['pred_shape_idx'][idx, cat_idx_mask[:top_k].view(-1)]
            best_shape_idx[idx] = best_shape_idx_topk[idx, 0]
    else:
        best_shape_idx_topk = predictions['pred_shape_idx'][:, :top_k]
        best_shape_idx = predictions['pred_shape_idx'][:, 0]
    
    cat_match = (best_cat_idx == gt_cat_idx).float()
    shape_match = (best_shape_idx == gt_shape_idx).float()
    shape_match_topk = torch.any(best_shape_idx_topk == gt_shape_idx.view(-1, 1), dim=-1).float()
    
    if cfg.eval.metrics.use_lfd:
        lfd_l1s = []
        with h5py.File(cfg.data.lfd_path, 'r') as h5:
            for idx, gt_shape_idx in enumerate(tqdm(predictions["gt_shape_idx"])):
                gt_lfd = h5['lfd'][gt_shape_idx.item()]
                pred_lfd = h5['lfd'][best_shape_idx[idx]]
                lfd_l1 = np.abs(pred_lfd - gt_lfd).sum(-1).mean(-1)
                lfd_l1s.append(lfd_l1)
        lfd_l1s = torch.tensor(lfd_l1s).to(pl_module.device)
    
    eval_res = {'micro': {'Acc@1': shape_match.mean().item(),
                          f'Acc@{top_k}': shape_match_topk.mean().item(),
                          'CatAcc': cat_match.mean().item(),
                          'LFD': lfd_l1s.mean().item(),
                          },
                'macro': {'Acc@1': 0,
                          f'Acc@{top_k}': 0,
                          'CatAcc': 0,
                          'LFD': 0,
                          }
                }
        
    cat_list = torch.unique(predictions["gt_cat_idx"])
    for cat_idx in cat_list:
        cat = IDX_TO_CLASS[cat_idx.item()]
        selected_idx = torch.nonzero(predictions["gt_cat_idx"] == cat_idx).view(-1)
        eval_res[cat] = {'Acc@1': shape_match[selected_idx].mean().item(), 
                         f'Acc@{top_k}': shape_match_topk[selected_idx].mean().item(),
                         'CatAcc': cat_match[selected_idx].mean().item(),
                         'LFD': lfd_l1s[selected_idx].mean().item(), 
                         }
        for key in eval_res[cat]:
            eval_res['macro'][key] += eval_res[cat][key]
    
    for key in eval_res['macro']:
        eval_res['macro'][key] /= len(cat_list)
    
    for key, value in eval_res['micro'].items():
        pl_module.log("val/micro_{}".format(key), value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
    log_cols = ['val', *list(eval_res['micro'].keys())]
    log_data = []
    for key in eval_res:
        log_data.append([key, *list(eval_res[key].values())])
    pl_module.logger.log_text(key=f"{pl_module.current_epoch}", columns=log_cols, data=log_data)



def compute_full_metrics(predictions, cfg, top_k=5, restrict_cat=False, device='cpu'):
    if cfg.data.name != cfg.data.shape_feats_source:
        obj_path = os.path.join(f"./data", cfg.data.shape_feats_source, f"{cfg.data.shape_feats_source}_obj.h5")
    else:
        obj_path = cfg.data.obj_path
    with h5py.File(obj_path, 'r') as h5:
        all_obj_points = h5['obj_points'][:]
        all_obj_normals = h5['obj_normals'][:]
    if cfg.data.extra_shapes:
        for shape_source in cfg.data.extra_shapes:
            with h5py.File(os.path.join(f"./data", shape_source, f"{shape_source}_obj.h5"), 'r') as h5:
                all_obj_points = np.concatenate([all_obj_points, h5['obj_points'][:]], axis=0)
                all_obj_normals = np.concatenate([all_obj_normals, h5['obj_normals'][:]], axis=0)
    if restrict_cat:
        predictions['cat_idx_mask'] = []
    num = len(predictions['gt_shape_idx'])
    metrics = {}
    if cfg.eval.metrics.use_rr: metrics['mRR'] = torch.zeros((num, 1), dtype=torch.float32, device=device)

    best_shape_idx_topk = torch.zeros_like(predictions['pred_shape_idx'][:, :top_k])
    best_shape_idx = torch.zeros_like(predictions['pred_shape_idx'][:, 0])
    
    for idx, cat_idx in enumerate(predictions["gt_cat_idx"]):
        torch.cuda.empty_cache()
        if restrict_cat:
            cat_idx_mask = torch.nonzero(predictions['pred_cat_idx'][idx].view(-1) == cat_idx)
            predictions['cat_idx_mask'].append(cat_idx_mask)
            best_shape_idx_topk[idx] = predictions['pred_shape_idx'][idx, cat_idx_mask[:top_k].view(-1)]
            best_shape_idx[idx] = best_shape_idx_topk[idx, 0]
        else:
            best_shape_idx_topk[idx] = predictions['pred_shape_idx'][idx, :top_k]
            best_shape_idx[idx] = predictions['pred_shape_idx'][idx, 0]
        
        if cfg.eval.metrics.use_rr:
            if restrict_cat:
                pred_shape_indices = predictions['pred_shape_idx'][idx, cat_idx_mask.view(-1)]
            else:
                pred_shape_indices = predictions['pred_shape_idx'][idx, :]
            metrics['mRR'][idx] = 1 / ((pred_shape_indices == gt_shape_idx).nonzero(as_tuple=False)[0].item() + 1)
    
    if cfg.eval.metrics.use_acc:
        gt_cat_idx = predictions['gt_cat_idx']
        best_cat_idx = predictions['pred_cat_idx'][:, 0]
        gt_shape_idx = predictions['gt_shape_idx']
        metrics['Acc@1'] = (best_shape_idx == gt_shape_idx).float().view(-1, 1)
        metrics[f'Acc@{top_k}'] = torch.any(best_shape_idx_topk == gt_shape_idx.view(-1, 1), dim=-1).float().view(-1, 1)
        metrics['CatAcc'] = (best_cat_idx == gt_cat_idx).float().view(-1, 1)
    
    print("compare meshes...")
    all_pc_metric_opts = [('CD', cfg.eval.metrics.use_cd), ('NC', cfg.eval.metrics.use_nc), ('F1@0.1', cfg.eval.metrics.use_f1), ('F1@0.3', cfg.eval.metrics.use_f1), ('F1@0.5', cfg.eval.metrics.use_f1)]
    pc_metrics = {m: torch.zeros((num, top_k), dtype=torch.float32, device=device) for m, use_flag in all_pc_metric_opts if use_flag}
    for idx, cat_idx in enumerate(tqdm(predictions["gt_cat_idx"])):
        best_shapes = [[torch.from_numpy(all_obj_points[bs_idx]), torch.from_numpy(all_obj_normals[bs_idx])] for bs_idx in best_shape_idx_topk[idx]]
        with h5py.File(cfg.data.obj_path, 'r') as gt_h5:
            gt_shape_idx = predictions['gt_shape_idx'][idx]
            gt_shape = [torch.from_numpy(gt_h5['obj_points'][gt_shape_idx]), torch.from_numpy(gt_h5['obj_normals'][gt_shape_idx])]
        shape_metrics = compute_shape_scores(gt_shape, best_shapes, device, thresholds=[0.1, 0.3, 0.5]) # compute reconstruction metrics
        for mn in pc_metrics:
            pc_metrics[mn][idx] = shape_metrics[mn]
    for mn in pc_metrics:
        metrics[mn] = pc_metrics[mn]
        metrics[f'{mn}@{top_k}'] = pc_metrics[mn].mean(dim=-1, keepdim=True)
    
    if cfg.eval.metrics.use_lfd:
        if cfg.data.name != cfg.data.shape_feats_source:
            lfd_path = os.path.join(f"./data", cfg.data.shape_feats_source, f"lfd_200.h5")
        else:
            lfd_path = cfg.data.lfd_path
        with h5py.File(lfd_path, 'r') as h5:
            all_lfds = h5['lfd'][:]
        if cfg.data.extra_shapes:
            for shape_source in cfg.data.extra_shapes:
                with h5py.File(os.path.join(f"./data", shape_source, f"lfd_200.h5"), 'r') as h5:
                    all_lfds = np.concatenate([all_lfds, h5['lfd'][:]], axis=0)
        all_lfd_l1s = []
        with h5py.File(cfg.data.lfd_path, 'r') as gt_h5:
            for idx, gt_shape_idx in enumerate(tqdm(predictions["gt_shape_idx"])):
                gt_lfd = gt_h5['lfd'][gt_shape_idx.item()]
                preds, indices = best_shape_idx_topk[idx].sort()
                pred_lfds = all_lfds[preds.cpu().numpy()] # indices have to be sorted for h5 indexing
                lfd_l1s = np.abs(pred_lfds - gt_lfd[None, ...]).sum(-1).mean(-1)
                lfd_l1s = torch.gather(torch.from_numpy(lfd_l1s).to(device), 0, indices.argsort())
                all_lfd_l1s.append(lfd_l1s.unsqueeze(0))
        all_lfd_l1s = torch.cat(all_lfd_l1s, dim=0)
        metrics['LFD'] = all_lfd_l1s
        metrics[f'LFD@{top_k}'] = all_lfd_l1s.mean(dim=-1, keepdim=True)
    
    if cfg.eval.metrics.use_mask_iou:
        all_mask_ious, all_lfd_l1s, all_lpips = evaluate_render_view(predictions, best_shape_idx_topk, cfg, max_size=512 if cfg.data.name=='pix3d' else 1024, device=device)
        metrics['MIoU'] = all_mask_ious#[:, 0].view(-1, 1)
        metrics[f'MIoU@{top_k}'] = all_mask_ious.mean(dim=-1, keepdim=True)
        metrics['vLFD'] = all_lfd_l1s#[:, 0].view(-1, 1)
        metrics[f'vLFD@{top_k}'] = all_lfd_l1s.mean(dim=-1, keepdim=True)
        metrics['LPIPS'] = all_lpips#[:, 0].view(-1, 1)
        metrics[f'LPIPS@{top_k}'] = all_lpips.mean(dim=-1, keepdim=True)

    if cfg.eval.metrics.use_ap_mesh:
        print("compute ap mesh...")
        metrics['APmesh'] = compute_ap_mesh(metrics['F1@0.3'][:, 0], predictions, restrict_cat=restrict_cat, device=device)
    
    return metrics


def compute_ap_mesh(f1_scores, predictions, f1_thres=0.5, restrict_cat=False, device='cpu'):
    ap_mesh_input = {
        'mesh_covered': [],
        'ap_mesh_scores': [torch.tensor([], dtype=torch.float32, device=device)],
        'ap_mesh_labels': [torch.tensor([], dtype=torch.uint8, device=device)],
    }
    
    for idx, cat_idx in enumerate(predictions["gt_cat_idx"]):
        pred_f1 = f1_scores[idx].item() / 100.0
        if restrict_cat:
            cat_idx_mask = predictions['cat_idx_mask'][idx]
            pred_score = predictions['sim_scores_sorted'][idx, cat_idx_mask[0]]
        else:
            pred_score = predictions['sim_scores_sorted'][idx, [0]]
        tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
        if (pred_f1 > f1_thres) and (idx not in ap_mesh_input['mesh_covered']):
            tpfp[0] = 1
            ap_mesh_input['mesh_covered'].append(idx)
        ap_mesh_input['ap_mesh_scores'].append(pred_score)
        ap_mesh_input['ap_mesh_labels'].append(tpfp)
    
    ap_mesh_input['ap_mesh_scores'] = torch.cat(ap_mesh_input['ap_mesh_scores'])
    ap_mesh_input['ap_mesh_labels'] = torch.cat(ap_mesh_input['ap_mesh_labels'])
    
    ap_mesh = {}
    ap_mesh['all'] = compute_ap(ap_mesh_input['ap_mesh_scores'], 
                                      ap_mesh_input['ap_mesh_labels'], 
                                      len(f1_scores)).item()
    cat_list = torch.unique(predictions["gt_cat_idx"])
    for cat_idx in cat_list:
        cat = IDX_TO_CLASS[cat_idx.item()]
        selected_idx = torch.nonzero(predictions["gt_cat_idx"] == cat_idx).view(-1)
        ap_mesh[cat] = compute_ap(ap_mesh_input['ap_mesh_scores'][selected_idx], 
                                        ap_mesh_input['ap_mesh_labels'][selected_idx], 
                                        len(selected_idx)).item()
    
    return ap_mesh


def evaluate_retrieval_offline(predictions, pl_module, restrict_cat=True, save_metrics=True):
    pl_module.text_logger.info(f"==>> Evaluation at Epoch {pl_module.current_epoch}")
    cfg = pl_module.cfg
    device = pl_module.device
    
    metrics = compute_full_metrics(predictions, cfg, restrict_cat=restrict_cat, device=device)
    if save_metrics:
        save_all_metrics(metrics, predictions, pl_module, restrict_cat=True)

    eval_res = {'micro': {m: vs.mean(0)[0].item() for m, vs in metrics.items() if m != 'APmesh'},
                'macro': {m: 0 for m in metrics.keys()}}
    if cfg.eval.metrics.use_ap_mesh:
        eval_res['micro']['APmesh'] = metrics['APmesh']['all']
    
    pl_module.text_logger.info(f"""Micro\n{"".join([f"{key:>10}:{eval_res['micro'][key]:>10.4f}{chr(10)}" for key in eval_res['micro']])}""")
    
    cat_list = torch.unique(predictions["gt_cat_idx"])
    for cat_idx in cat_list:
        cat = IDX_TO_CLASS[cat_idx.item()]
        selected_idx = torch.nonzero(predictions["gt_cat_idx"] == cat_idx).view(-1)
        eval_res[cat] = {m: vs[selected_idx].mean(0)[0].item() for m, vs in metrics.items() if m != 'APmesh'}
        if cfg.eval.metrics.use_ap_mesh:
            eval_res[cat]['APmesh'] = metrics['APmesh'][cat]
        pl_module.text_logger.debug(f"""{cat}\n{"".join([f"{key:>10}:{eval_res[cat][key]:>10.4f}{chr(10)}" for key in eval_res[cat]])}""")
        for key in eval_res[cat]:
            eval_res['macro'][key] += eval_res[cat][key]
    
    for key in eval_res['macro']:
        eval_res['macro'][key] /= len(cat_list)
    pl_module.text_logger.info(f"""Macro\n{"".join([f"{key:>10}:{eval_res['macro'][key]:>10.4f}{chr(10)}" for key in eval_res['macro']])}""")
    
    ### LaTex format
    for cat_idx in cat_list:
        cat = IDX_TO_CLASS[cat_idx.item()]
        pl_module.text_logger.debug(f"""{cat} & {" & ".join([f"{eval_res[cat][key]:.4f}" for key in eval_res[cat]])}""")
    pl_module.text_logger.info(f"""micro & {" & ".join([f"{eval_res['micro'][key]:.4f}" for key in eval_res['micro']])}""")
    pl_module.text_logger.info(f"""macro & {" & ".join([f"{eval_res['macro'][key]:.4f}" for key in eval_res['macro']])}""")
