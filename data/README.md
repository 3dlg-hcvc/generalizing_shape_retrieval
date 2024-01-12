## Pix3D

The data files should be organized as follows:
```sh
gcmic
├── data
│   ├── pix3d
│   │   ├── data # pix3d raw data
│   │   │   ├── img
│   │   │   ├── mask
│   │   │   ├── ...
│   │   ├── mask2former # predicted object mask using Mask2Former
│   │   ├── pix3d_annotation_all.txt
│   │   ├── pix3d_annotation_easy.txt # annotation file containing object queries w/o occlusions
│   │   ├── pix3d_annotation_hard.txt # annotation file containing object queries w/ occlusions
│   │   ├── pix3d_224.h5 # image queries
│   │   ├── pix3d_mv.h5 # multiviews for each shape
│   │   ├── pix3d_obj.h5 # pointcloud for each shape
│   │   ├── lfd_200.h5 # 200-view LFD for each shape
│   │   ├── pix3d_pose.json # object pose info for rendering
```

Please refer to `../preprocess/pix3d/gen_dataset_hdf5.py`, `../preprocess/pix3d/get_all_lfd.py` and `../preprocess/pix3d/extract_pose_json.py` for how to prepare preprocessed data.

## Scan2CAD

The data files should be organized as follows:
```sh
gcmic
├── data
│   ├── scan2cad
│   │   ├── Images
│   │   ├── Rendering
│   │   ├── mask # object instance mask
│   │   ├── scan2cad_annotation.txt
│   │   ├── scan2cad_480x360.h5 # image queries
│   │   ├── scan2cad_mv.h5 # multiviews for each shape
│   │   ├── scan2cad_obj.h5 # pointcloud for each shape
│   │   ├── lfd_200.h5 # 200-view LFD for each shape
│   │   ├── scan2cad_pose.json # object pose info for rendering
```

Please refer to `../preprocess/scan2cad/gen_dataset_hdf5.py`, `../preprocess/scan2cad/get_all_lfd.py` and `../preprocess/scan2cad/extract_pose_json.py` for how to prepare preprocessed data.