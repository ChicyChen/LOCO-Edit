from PIL import Image
from torch.utils.data import Dataset
import pickle 
import os
import numpy as np
import torch
import torchvision.transforms as tfs

class CelebAMaskDataLoader(Dataset):
    def __init__(self, root, save_path = None, res = 256):
        self.root = root
        print(self.root)
        self.idx2sem = {
            0: 'background',	
            1: 'skin',	
            2: 'nose',
            3: 'eye_g',	
            4: 'l_eye',	
            5: 'r_eye',
            6: 'l_brow',	
            7: 'r_brow',	
            8: 'l_ear',
            9: 'r_ear',	
            10: 'mouth',	
            11: 'u_lip',
            12: 'l_lip',	
            13: 'hair',	
            14: 'hat',
            15: 'ear_r',	
            16: 'neck_l',	
            17: 'neck',
            18: 'cloth',
        }
        self.sem2idx = {v: k for k, v in self.idx2sem.items()}
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.dataset = self._parse_CelebAMask()
        self.res = res
        self.transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])


    def _parse_CelebAMask(self):

        self.celebA_path = os.path.join(self.root, "CelebA-HQ-img")
        self.celebAMask_path = os.path.join(self.root, "CelebAMask-HQ-mask-anno")
        dataset = os.listdir(self.celebA_path)
        dataset.sort()
        datasetdir = {}
        for file in dataset:
            idx= int(file.split(".")[0])
            datasetdir[idx] = {
                "image_path" : os.path.join(self.celebA_path, file),
                "annotation_path" : [],
            }
        subdirs = os.listdir(self.celebAMask_path)
        subdirs.sort()
        for subdir in subdirs:
            if subdir.isdigit():
                dir = os.path.join(self.celebAMask_path, subdir)
                annotaion_files = os.listdir(dir)
                annotaion_files.sort()
                for f in annotaion_files:
                    if f[:5].isdigit():
                        annotate_idx = int(f[:5])
                        datasetdir[annotate_idx]["annotation_path"].append(os.path.join(dir, f))
         
        
        return datasetdir
    
    def __len__(self):
        return len(self.dataset.keys())
    
    
    def __getitem__(self, idx):
        img_path = self.dataset[idx]["image_path"]
        img = Image.open(img_path).resize((self.res, self.res))


        if self.save_path is not None:
            img.save(os.path.join(self.save_path, f"image_{idx}.png"))

        return self.transform(img).unsqueeze(0)
    
    def getmask(self, idx, choose_sem, list_sem = True):

        img_path = self.dataset[idx]["image_path"]
        img_mask_paths = self.dataset[idx]["annotation_path"]
        img = Image.open(img_path).resize((self.res, self.res))
        image_np = np.array(img)
    
        semlist = []
        for img_mask_path in img_mask_paths:
            sem = os.path.basename(img_mask_path)[6:].split(".")[0]
            assert sem in self.sem2idx.keys()
            semlist.append(sem)
        if list_sem:
            print(f"Available semantics are {semlist}")
        assert choose_sem in semlist, f"For the {idx}th image, choose semantic should be in the list {semlist}" 

        for img_mask_path in img_mask_paths:
            sem = os.path.basename(img_mask_path)[6:].split(".")[0]
            if sem == choose_sem:
                mask = Image.open(img_mask_path).resize((self.res, self.res))
                mask_np = np.array(mask)
                break

        if self.save_path is not None:
            mask.save(os.path.join(self.save_path, f"image_{idx}_mask_{choose_sem}.png"))
            img_mask_mp = image_np.copy()
            img_mask_mp_neg = image_np.copy()
            # img_mask_mp[mask_np.astype(bool)] = (image_np[mask_np.astype(bool)]/2 + 64) 
            # img_mask_mp_neg[~mask_np.astype(bool)] = (image_np[mask_np.astype(bool)]/2 + 64) 
            img_mask_mp[mask_np.astype(bool)] = (image_np[mask_np.astype(bool)]*0) 
            img_mask_mp_neg[~mask_np.astype(bool)] = (image_np[~mask_np.astype(bool)]*0) 
            Image.fromarray(img_mask_mp).save(os.path.join(self.save_path, f"demo_image_{idx}_mask_{choose_sem}.png"))
            Image.fromarray(img_mask_mp_neg).save(os.path.join(self.save_path, f"demo_image_{idx}_neg_mask_{choose_sem}.png"))


        return torch.tensor(mask_np.astype(bool)).permute(2, 0, 1)


if __name__ == "__main__":
    import random
    dataloader = CelebAMaskDataLoader(
                    root = "/scratch/qingqu_root/qingqu1/shared_data/celebA-HQ-mask/CelebAMask-HQ",
                    save_path = "/scratch/qingqu_root/qingqu1/huijiezh/cgen_part/src/datasets")
    idx = random.randint(0, len(dataloader))
    dataloader[idx]
    dataloader.getmask(idx, choose_sem="hair", list_sem = True)
