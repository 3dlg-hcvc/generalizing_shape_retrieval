# Generalizing Single-View 3D Shape Retrieval to Occlusions and Unseen Objects

Qirui Wu, [Daniel Ritchie](https://dritchie.github.io/), [Manolis Savva](https://msavva.github.io/), [Angel Xuan Chang](http://angelxuanchang.github.io/)

[[arXiv](https://arxiv.org/abs/2401.00405), [Project Page](https://3dlg-hcvc.github.io/generalizing_shape_retrieval/), [Dataset](https://github.com/3dlg-hcvc/generalizing_shape_retrieval#data)]

<!-- ![](docs/images/teaser.png) -->
<p><img src="docs/images/teaser.png" width="65%"></p>

Official repository of the paper [Generalizing Single-View 3D Shape Retrieval to Occlusions and Unseen Objects](https://github.com/3dlg-hcvc/generalizing_shape_retrieval). We systematically study the generalization of single-view 3D shape retrieval along three different axes: the presence of object occlusions and truncations, generalization to unseen 3D shape data, and generalization to unseen objects in the input images.


## Setup
The environment is tested with Python 3.8, PyTorch 2.0, CUDA 11.7, PyTorch3D 0.7.3, Lightning 2.0.1.

```bash
conda create -n gcmic python=3.8
conda activate gcmic
pip3 install torch torchvision
pip install -r requirements.txt
conda install -c fvcore -c iopath -c bottler -c conda-forge fvcore iopath nvidiacub
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.3"
```


## Data

### MOOS

<p><img src="docs/images/moos_generation.png" width="100%"></p>

Multi-Object Occlusion Scenes (MOOS) is generated using a heuristic algorithm that iteratively places newly sampled 3D shapes from [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future) into the existing layout. Download MOOS **raw** and **preprocessed** data with the following command and extract/place them at `./data/moos`.
```sh
cd data/moos 
wget https://aspis.cmpt.sfu.ca/projects/gcmic/data/moos/scenes.zip && unzip scenes.zip
wget https://aspis.cmpt.sfu.ca/projects/gcmic/data/moos/moos_annotations.zip && unzip moos_annotations.zip
wget https://aspis.cmpt.sfu.ca/projects/gcmic/data/moos/moos_h5.tar && tar -xvf moos_h5.tar
```
The data files should be organized as follows:
```shell
gcmic
├── data
│   ├── moos
│   │   ├── scenes # raw image data
│   │   │   ├── <scene_name>
│   │   │   │   ├── rgb
│   │   │   │   │   ├── rgb_<view_id>.rgb.png
│   │   │   │   ├── instances
│   │   │   │   │   ├── instances_<view_id>.rgb.png
│   │   │   │   ├── objects
│   │   │   │   │   ├── <obj_id>_<view_id>.rgb.png
│   │   │   │   │   ├── <obj_id>_<view_id>.mask.png
│   │   │   │   ├── depth
│   │   │   │   ├── normal
│   │   │   │   ├── layout2d.png # top-down view
│   │   │   │   ├── scene.json # scene metadata
│   │   ├── moos_annotation.txt
│   │   ├── moos_annotation_all.txt
│   │   ├── moos_annotation_no_occ.txt # annotation file containing object queries w/o occlusions
│   │   ├── moos_annotation_occ.txt # annotation file containing object queries w/ occlusions
│   │   ├── moos_1k.h5 # image queries
│   │   ├── moos_mv.h5 # multiviews for each shape
│   │   ├── moos_obj.h5 # pointcloud for each shape
│   │   ├── lfd_200.h5 # 200-view LFD for each shape
│   │   ├── moos_pose.json # object pose info for rendering
│   │   ├── ...
```

Please refer to `./preprocess/moos/gen_dataset_hdf5.py`, `./preprocess/3dfuture/get_all_lfd.py` and `./preprocess/moos/extract_pose_json.py` for how to prepare preprocessed data. Please refer to [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future) for downloading 3D shapes if you want to render your own shape multiviews and LFDs. Put 3D-FUTURE data under `./data/3dfuture`.

We generate 10K scenes with the script `./preprocess/moos/render_scenes.py`. Also note that we can reconstruct each scene by reading meta information from `scene.json` (run `./preprocess/moos/reconstruct_scenes.py`). You can explore more demos of how to generate random scenes in `./notebook`.

### Pix3D

Download Pix3D raw data [here](http://pix3d.csail.mit.edu/), and preprocessed data with the following command and extract/place them at `./data/pix3d`.
```sh
cd data/pix3d
wget https://aspis.cmpt.sfu.ca/projects/gcmic/data/pix3d/mask2former.zip && unzip mask2former.zip
wget https://aspis.cmpt.sfu.ca/projects/gcmic/data/pix3d/pix3d_annotations.zip && unzip pix3d_annotations.zip
wget https://aspis.cmpt.sfu.ca/projects/gcmic/data/pix3d/pix3d_h5.tar && tar -xvf pix3d_h5.tar
```
Please refer to [details](./data/README.md#pix3d) for Pix3D data structure.

### Scan2CAD

Download ScanNet25K images and CAD annotations from [ROCA data](https://github.com/cangumeli/ROCA#downloading-processed-data-recommended), and preprocessed data with the following command and extract/place them at `./data/scan2cad`.
```sh
cd data/scan2cad
wget https://aspis.cmpt.sfu.ca/projects/gcmic/data/scan2cad/mask.zip && unzip mask.zip
wget https://aspis.cmpt.sfu.ca/projects/gcmic/data/scan2cad/scan2cad_annotations.zip && unzip scan2cad_annotations.zip
wget https://aspis.cmpt.sfu.ca/projects/gcmic/data/scan2cad/scan2cad_h5.tar && tar -xvf scan2cad_h5.tar
```
Please refer to [details](./data/README.md#scan2cad) for Scan2CAD data structure. Download ShapeNet 3D shapes [here](https://shapenet.org/) if you want to render your own shape multiviews and LFDs.


## Train

Train a CMIC model on the ALL set of MOOS.
```sh
python train.py -t train -e cmic_moos --data_conf conf/dataset/moos.yaml --model_conf conf/model/cmic.yaml --epochs 50 --batch_size 64 --num_views 12 --verbose False --annotation_file moos_annotation_all.txt --use_crop --use_1k_img
```

Train a CMIC model on the ALL set of Pix3D using Mask2Former predicted object masks.
```sh
python train.py -t train -e cmic_pix3d --data_conf conf/dataset/pix3d.yaml --model_conf conf/model/cmic.yaml --epochs 500 --batch_size 64 --num_views 12 --verbose False --annotation_file pix3d_annotation_all.txt --mask_source m2f_mask --val_check_interval 1 --use_crop
```

Train a CMIC model on Scan2CAD.
```sh
python train.py -t train -e cmic_scan2cad --data_conf conf/dataset/scan2cad.yaml --model_conf conf/model/cmic.yaml --epochs 500 --batch_size 64 --num_views 12 --verbose False --annotation_file scan2cad_annotation.txt --val_check_interval 1 --num_sanity_val_steps 100 --use_crop --use_480p_img --center_in_image
```


## Fine-tune

Fine-tune `cmic_moos` on Pix3D
```sh
python train.py -t finetune -e cmic_moos_ft_pix3d --data_conf conf/dataset/pix3d.yaml --model_conf conf/model/cmic.yaml --epochs 5 --batch_size 64 --num_views 12 --verbose False --annotation_file pix3d_annotation_all.txt --mask_source m2f_mask --ckpt_path ./output/moos/cmic/cmic_moos/train/model.ckpt --val_check_interval 1 --use_crop
```

Fine-tune `cmic_moos` on Scan2CAD
```sh
python train.py -t finetune -e cmic_moos_ft_scan2cad --data_conf conf/dataset/scan2cad.yaml --model_conf conf/model/cmic.yaml --epochs 5 --batch_size 64 --num_views 12 --verbose False --annotation_file scan2cad_annotation.txt --ckpt_path ./output/moos/cmic/cmic_moos/train/model.ckpt --val_check_interval 1 --use_crop --num_sanity_val_steps 400 --use_480p_img --center_in_image
```


## Evaluation

We first embed all shape multiviews from different datasets (MOOS, Pix3D, and Scan2CAD) using the specified pretrained shape encoder.
```sh
python test.py -t embed_shape -e <model_name> --data_conf conf/dataset/<dataset>.yaml --model_conf conf/model/cmic.yaml --batch_size 48 --num_views 12 --verbose False --ckpt model.ckpt 
```

Evaluate on `all|seen|unseen` objects of different **MOOS** sets `all|no_occ|occ`.
```sh
python test.py -t test -e cmic_moos --data_conf conf/dataset/moos.yaml --model_conf conf/model/cmic.yaml --verbose False --batch_size 48 --ckpt model.ckpt --annotation_file moos_annotation_<all|no_occ|occ>.txt --offline_evaluation --test_objects <all|seen|unseen> --use_crop --use_1k_img
```

Evaluate on `all|seen|unseen` objects of different **Pix3D** sets `all|easy|hard`.
```sh
python test.py -t test -e cmic_pix3d --data_conf conf/dataset/pix3d.yaml --model_conf conf/model/cmic.yaml --verbose False --batch_size 48 --mask_source m2f_mask --ckpt model.ckpt --annotation_file pix3d_annotation_<all|easy|hard>.txt --offline_evaluation --test_objects <all|seen|unseen> --use_crop
```

Evaluate on the **Scan2CAD** dataset.
```sh
python test.py -t test -e cmic_scan2cad --data_conf conf/dataset/scan2cad.yaml --model_conf conf/model/cmic.yaml --verbose False --batch_size 48 --ckpt model.ckpt --annotation_file scan2cad_annotation.txt --offline_evaluation --use_crop --use_480p_img
```

**Note**
- Add flags `--not_eval_acc --shape_feats_source <dataset>` to test on unseen 3D shapes.
- Add flag `--save_eval_vis` to save retrieved 3D shape renderings and visualizations.

## Acknowledgement

We thank [Lewis Lin](https://github.com/LewisLinn) for helping developing the metric code in the early stage of the project.


## Bibtex
```
@article{wu2023generalizing,
    author  = {Wu, Qirui and Ritchie, Daniel and Savva, Manolis and Chang, Angel.X},
    title   = {{Generalizing Single-View 3D Shape Retrieval to Occlusions and Unseen Objects}},
    year    = {2023},
    eprint  = {2401.00405},
    archivePrefix   = {arXiv},
    primaryClass    = {cs.CV}
}
```

