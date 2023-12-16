import h5py
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

cat2synsetId = {'chair': '03001627', 'bed': '02818832', 'table': '04379243', 'sofa': '04256520'}
synsetId2cat = {v:k for k,v in cat2synsetId.items()}
CLASS_TO_IDX = {"chair": 0, "bed": 1, "sofa": 2,"table": 3}
IDX_TO_CLASS = {v:k for k,v in CLASS_TO_IDX.items()}


class Scan2CADShape(Dataset):
    def __init__(self, cfg):
        super(Scan2CADShape, self).__init__()
        self.cfg = cfg
        
        self.data_dir = cfg.data.preprocessed_path
        self.mv_path = cfg.data.mv_path
        self.task = cfg.general.task
        
        self.input_dim = cfg.data.input_dim
        self.prepare_dataset()

        if self.task == 'trian':
            self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=(self.input_dim, self.input_dim), scale=(0.65, 0.9)), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                ])

    def prepare_dataset(self):        
        with h5py.File(self.mv_path, 'r') as h5:
            self.obj_names = [name.decode("utf8") for name in h5['obj_names'][:]]

    def __getitem__(self, idx):
        obj_name = self.obj_names[idx]
        obj_id = f"shapenet.{obj_name}"
        synsetId = obj_name.split('/')[0]
        cat = synsetId2cat[synsetId]
        cat_idx = CLASS_TO_IDX[cat]
        
        with h5py.File(self.mv_path, 'r') as h5:
            multiview = h5['multiview'][idx] # (12, 224, 224, 3)
            multiview = torch.cat([self.transform(Image.fromarray(multiview[mv_id])).unsqueeze(0) for mv_id in range(len(multiview))], 0)

        return {'shape': multiview, 'cat_idx': cat_idx, 'shape_idx': idx, 'shape_id': obj_id}

    def __len__(self):
        return len(self.obj_names)
    

if __name__ == '__main__':
    import time
    from omegaconf import OmegaConf
    conf_list = ['conf/default.yaml', 'conf/dataset/pix3d.yaml', 'conf/model/cmic.yaml']
    output_dir = 'output'
    cfg = OmegaConf.merge(*[OmegaConf.load(f) for f in conf_list])
    
    shape_dataset = Scan2CADShape(cfg, 'train')
    shape_dataloader = DataLoader(shape_dataset, batch_size=cfg.data.batch_size, num_workers=1, drop_last=True, pin_memory=True)
    shape_iter = iter(shape_dataloader)
    batch = next(shape_iter)
    import pdb; pdb.set_trace()