import os
import argparse
from omegaconf import OmegaConf


def load_cfg():
    parser = argparse.ArgumentParser()
    
    # general
    parser.add_argument('-t', '--task', type=str, default='', help='train | test')
    parser.add_argument('-e', '--experiment', type=str, default='', help='specify experiment')
    parser.add_argument('-n', '--num_nodes', type=int, default=1, help='specify num of gpu nodes')
    parser.add_argument('-s', '--split', type=str, default='val', help='specify data split')
    parser.add_argument('--data_conf', type=str, default='conf/dataset/pix3d.yaml', help='specify data conf yaml file')
    parser.add_argument('--model_conf', type=str, default='conf/model/cmic.yaml', help='specify model conf yaml file')
    parser.add_argument('--epochs', type=int, default=500, help='specify training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='specify batch_size')
    parser.add_argument('--optimizer', type=str, default='Adam', help='specify optimizer')
    parser.add_argument('--lr', type=float, default=None, help='specify initial learning rate')
    parser.add_argument('--ckpt', type=str, default=None, help='what checkpoint to use')
    parser.add_argument('--ckpt_path', type=str, default=None, help='specify checkpoint path')
    parser.add_argument('--num_sanity_val_steps', type=int, default=10, help='specify number of sanity check steps on val')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=None, help='specify number of epochs between validation')
    parser.add_argument('--val_check_interval', type=float, default=None, help='specify number of step between validation')
    parser.add_argument('--verbose', type=bool, default=True, help='whether log to console')
    parser.add_argument('--annotation_file', type=str, default=None, help='which annotation file to use')
    parser.add_argument('--img_source', type=str, default='image', help='which image source to use: image')
    parser.add_argument('--mask_source', type=str, default='mask', help='which mask source to use: mask | m2f_mask')
    parser.add_argument('--random_sample', type=int, default=None, help='sample subset of test set')
    parser.add_argument('--train_dataset', type=str, default=None, help='which dataset was used for training')
    parser.add_argument('--reduce_order_magnitude', type=int, default=None, help='reduce_order_magnitude')
    parser.add_argument('--test_only_occlusion', default=False, action="store_true", help='whether test only occluded masks')
    parser.add_argument('--center_in_image', default=False, action="store_true", help='whether use only center_in_image for scan2cad')
    parser.add_argument('--offline_evaluation', default=False, action="store_true", help='whether use shape features computed offline')
    parser.add_argument('--not_eval_acc', default=False, action="store_true", help='whether use accuracy metric')
    parser.add_argument('--not_eval_lfd', default=False, action="store_true", help='whether use lfd metric')
    parser.add_argument('--not_eval_view', default=False, action="store_true", help='whether use view dependent metrics')
    parser.add_argument('--save_eval_vis', default=False, action="store_true", help='whether save evaluation intermediate results')
    
    # CMIC
    parser.add_argument('--use_crop', default=False, action="store_true", help='whether use crop image')
    parser.add_argument('--use_color_transfer', default=False, action="store_true", help='whether use color transfer')
    parser.add_argument('--shape_mv_file', type=str, default='', help='specify shape file')
    parser.add_argument('--shape_feats_source', type=str, default=None, help='specify shape feats source')
    parser.add_argument('--extra_shapes', type=str, default=None, nargs='+', help='specify shape feats file for offline evalaution')
    parser.add_argument('--use_1k_img', default=False, action="store_true", help='whether use moos 1024x1024 images')
    parser.add_argument('--use_480p_img', default=False, action="store_true", help='whether use scan2cad 480x360 images')
    parser.add_argument('--use_vit_backbone', default=False, action="store_true", help='whether use vit backbone')
    parser.add_argument('--use_r18_backbone', default=False, action="store_true", help='whether use r18 backbone')
    parser.add_argument('--use_attn_block', default=False, action="store_true", help='whether use attn block')
    parser.add_argument('--use_multihead_attn', default=False, action="store_true", help='whether use_multihead_attn')
    parser.add_argument('--use_paper_loss', default=False, action="store_true", help='whether use original cmic loss in paper')
    parser.add_argument('--num_views', type=int, default=12, help='specify num of multiviews')
    parser.add_argument('--test_objects', type=str, default='all', help='what objects to test: all | seen | unseen')
    
    args = parser.parse_args()
    
    cfg_list = ['conf/default.yaml', args.data_conf, args.model_conf]
    cfg_list = [os.path.join(os.getcwd(), c) for c in cfg_list]
    cfg = OmegaConf.merge(*[OmegaConf.load(f) for f in cfg_list])

    cfg.general.task = args.task
    cfg.general.experiment = args.experiment
    cfg.train.epochs = args.epochs
    cfg.train.optim.classname = args.optimizer
    if args.lr:
        cfg.train.optim.lr = args.lr
    cfg.train.num_sanity_val_steps = args.num_sanity_val_steps
    if args.check_val_every_n_epoch:
        cfg.train.check_val_every_n_epoch = args.check_val_every_n_epoch
        cfg.train.val_check_interval = 1.0
    if args.val_check_interval:
        cfg.train.val_check_interval = args.val_check_interval
        cfg.train.check_val_every_n_epoch = 1
    cfg.data.batch_size = args.batch_size
    if args.random_sample:
        cfg.data.num_random_sample = args.random_sample
    if args.annotation_file:
        cfg.data.annotation_file = args.annotation_file
    if args.task != "train":
        cfg.data.split = args.split

    cfg.model.use_checkpoint = args.ckpt
    cfg.model.ckpt_path = args.ckpt_path
    cfg.log.use_console_log = args.verbose

    cfg.general.dataset = cfg.data.name if not args.train_dataset else args.train_dataset
    root = os.path.join(cfg.OUTPUT_PATH, cfg.general.dataset, cfg.general.model, cfg.general.experiment, cfg.general.task)
    os.makedirs(root, exist_ok=True)
    cfg.general.root = root
    
    cfg = load_cmic_cfg(cfg, args)
        
    return cfg, args


def load_cmic_cfg(cfg, args):
    cfg.data.img_source = args.img_source
    cfg.data.mask_source = args.mask_source
    cfg.data.use_crop = args.use_crop
    cfg.data.use_color_transfer = args.use_color_transfer
    cfg.data.reduce_order_magnitude = args.reduce_order_magnitude
    cfg.data.test_only_occlusion = args.test_only_occlusion
    cfg.data.center_in_image = args.center_in_image
    cfg.data.test_objects = args.test_objects
    
    cfg.data.multiview.mv_num = args.num_views
    if args.shape_mv_file:
        cfg.data.mv_path = os.path.join(cfg.data.preprocessed_path, args.shape_mv_file)
    if args.shape_feats_source:
        if args.shape_feats_source != cfg.data.name: assert args.not_eval_acc
        cfg.data.shape_feats_source = args.shape_feats_source
    else:
        cfg.data.shape_feats_source = cfg.data.name
    cfg.data.extra_shapes = args.extra_shapes
    if cfg.data.name == "moos" and args.use_1k_img:
        cfg.data.h5_path = os.path.join(cfg.data.preprocessed_path, "moos_1k.h5")
    if cfg.data.name == "scan2cad" and args.use_480p_img:
        cfg.data.h5_path = os.path.join(cfg.data.preprocessed_path, "scan2cad_480x360.h5")
    
    cfg.model.use_vit_backbone = args.use_vit_backbone
    cfg.model.use_r18_backbone = args.use_r18_backbone
    cfg.model.use_attn_block = args.use_attn_block
    cfg.model.use_multihead_attn = args.use_multihead_attn
    cfg.model.use_paper_loss = args.use_paper_loss
    
    cfg.eval.metrics.use_acc = not args.not_eval_acc
    cfg.eval.metrics.use_lfd = not args.not_eval_lfd
    cfg.eval.metrics.use_mask_iou = not args.not_eval_view
    cfg.eval.metrics.use_single_lfd = not args.not_eval_view
    cfg.eval.metrics.use_lpips = not args.not_eval_view
    cfg.eval.save_vis = args.save_eval_vis
        
    if args.task == "test" and args.offline_evaluation:
        cfg.callbacks = [cb+'Offline' if cb.endswith('Evaluate') else cb for cb in cfg.callbacks]

    return cfg