import json
import os
import types
import time
import gc
from tqdm import tqdm

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, einsum

from PIL import Image

########
# path #
########
from configs.paths import (
    DATASET_PATHS,
    MODEL_PATHS,
)

####################
# uncond diffusion #
####################
from diffusers import DDIMScheduler, DDIMPipeline

from models.guided_diffusion.script_util import g_DDPM


def concatenate_pil_horizontally(pils):
    widths, heights = zip(*(i.size for i in pils))

    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in pils:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_image

def get_prependicualr_component(x, y):
    assert x.shape == y.shape
    return x - ((torch.mul(x, y).sum())/(torch.norm(y)**2)) * y


def get_custom_diffusion_scheduler(args):
    '''
    DDIM scheduler
    '''
    if args.use_yh_custom_scheduler:
        scheduler = YHCustomScheduler(args)

    elif 'HF' in args.model_name:
        scheduler = None

    else:
        scheduler = DDIMScheduler(
            num_train_timesteps     = args.config.diffusion.num_diffusion_timesteps,
            beta_start              = args.config.diffusion.beta_start,
            beta_end                = args.config.diffusion.beta_end,
            beta_schedule           = args.config.diffusion.beta_schedule,
            trained_betas           = None,
            clip_sample             = False, 
            set_alpha_to_one        = False,    # follow Stable-Diffusion setting. NOTE : need to verify this setting 
            steps_offset            = 1,        # follow Stable-Diffusion setting. NOTE : need to verify this setting
            prediction_type         = "epsilon",
        )

    return scheduler
    
def get_custom_diffusion_model(args):
    '''
    CelebA_HQ, LSUN, AFHQ

    TODO : IMAGENET
    '''
    # pretrained weight url
    if args.model_name == "CelebA_HQ":
        url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        init_ckpt = torch.hub.load_state_dict_from_url(
            url, map_location=args.device
        )
    elif args.model_name in ['LSUN_bedroom', 'LSUN_cat', 'LSUN_horse']:
        init_ckpt = torch.load(MODEL_PATHS[args.model_name], map_location=args.device)
    elif args.model_name in ["ImageNet256Uncond", "ImageNet64Uncond", "ImageNet256Cond", "ImageNet128Cond", "ImageNet64Cond", "CIFAR10Uncond"]:
        init_ckpt = torch.load(MODEL_PATHS[args.model_name], map_location=args.device)
    elif args.model_name == "CelebA_HQ_HF":
        model_id = "google/ddpm-ema-celebahq-256"
    elif args.model_name == "LSUN_church_HF":
        model_id = "google/ddpm-ema-church-256"
    elif args.model_name == "LSUN_bedroom_HF":
        model_id = "google/ddpm-ema-bedroom-256"
    elif args.model_name == "FFHQ_HF":
        model_id = "google/ncsnpp-ffhq-256"
    elif args.model_name in ["FFHQ_P2", "AFHQ_P2", "Flower_P2", "Cub_P2", "Metface_P2"]:
        init_ckpt = torch.load(MODEL_PATHS[args.model_name])
    else:
        raise ValueError()

    # load model and weight
    if args.model_name in ["CelebA_HQ"]:
        model = PullBackDDPM(args)
        model.learn_sigma = False
        model.load_state_dict(init_ckpt)

    elif args.model_name in ["FFHQ_P2", "AFHQ_P2", "Flower_P2", "Cub_P2", "Metface_P2"]:
        model = g_DDPM(args)
        model.learn_sigma = True
        model.load_state_dict(init_ckpt)

    elif args.model_name in ["ImageNet256Uncond", "ImageNet64Uncond", "CIFAR10Uncond", "ImageNet256Cond", "ImageNet128Cond", "ImageNet64Cond", "LSUN_bedroom", "LSUN_cat", "LSUN_horse"]:
        model = g_DDPM(args)
        model.learn_sigma = True
        model.load_state_dict(init_ckpt)

    elif args.model_name in ["CelebA_HQ_HF", "LSUN_bedroom_HF", "LSUN_church_HF", "FFHQ_HF"]:
        model = DDIMPipeline.from_pretrained(model_id) 
        model.unet.get_res    = types.MethodType(get_res_uncond, model.unet)
        model.enable_xformers_memory_efficient_attention()


    else:
        raise ValueError('Not implemented dataset')

    # load weight to model
    model = model.to(args.device)
    return model



####################
# T2I diffusion models #
####################
from diffusers import (
    StableDiffusionPipeline, 
    DDIMScheduler,
    DiffusionPipeline, 
)


def get_stable_diffusion_scheduler(args, scheduler):
    # monkey patch (replace scheduler for better inversion quality)
    if args.use_yh_custom_scheduler:
        scheduler.t_max = 999
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=args.device, dtype=args.dtype)
        scheduler.betas = scheduler.betas.to(device=args.device, dtype=args.dtype)
        scheduler.set_timesteps = types.MethodType(set_timesteps, scheduler)
        scheduler.step = types.MethodType(step, scheduler)
    else:
        pass
    return scheduler

def get_deepfloyd_if_scheduler(args, scheduler):
    # monkey patch (replace scheduler for better inversion quality)
    if args.use_yh_custom_scheduler:
        scheduler.t_max = 990
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=args.device, dtype=args.dtype)
        scheduler.betas = scheduler.betas.to(device=args.device, dtype=args.dtype)
        scheduler.set_timesteps = types.MethodType(set_timesteps, scheduler)
        scheduler.step = types.MethodType(step, scheduler)
        pass
    else:
        pass
    return scheduler

def set_timesteps(self, num_inferences, device=None, is_inversion=False):
    device = 'cpu' if device is None else device
    if is_inversion:
        seq = torch.linspace(0, 1, num_inferences, device=device) * self.t_max
        seq = seq + 1e-6
        seq_prev = torch.cat([torch.tensor([-1], device=device), seq[:-1]], dim = 0)
        self.timesteps = seq_prev[1:]
        self.timesteps_next = seq[1:]
    
    else:
        seq = torch.linspace(0, 1, num_inferences, device=device) * self.t_max
        seq_prev = torch.cat([torch.tensor([-1], device=device), seq[:-1]], dim = 0)
        self.timesteps = reversed(seq[1:])
        self.timesteps_next = reversed(seq_prev[1:])
    
def step(self, et, t, xt, eta=0.0, **kwargs):
    '''
    Notation
        - a : alpha / b : beta / e : epsilon
    '''
    t_idx   = self.timesteps.tolist().index(t)
    t_next  = self.timesteps_next[t_idx]

    # extract need parameters : at, at_next
    at = extract(self.alphas_cumprod, t, xt.shape)
    at_next = extract(self.alphas_cumprod, t_next, xt.shape)

    # DDIM step ; xt-1 = sqrt(at-1 / at) (xt - sqrt(1-at)*e(xt, t)) + sqrt(1-at-1)*e(xt, t)
    P_xt = (xt - et * (1 - at).sqrt()) / at.sqrt()
    # Deterministic. (ODE)
    if eta == 0:
        D_xt = (1 - at_next).sqrt() * et
        xt_next = at_next.sqrt() * P_xt + D_xt

    # Add noise. (SDE)
    else:
        sigma_t = ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()

        D_xt = (1 - at_next - eta * sigma_t ** 2).sqrt() * et
        xt_next = at_next.sqrt() * P_xt + D_xt + eta * sigma_t * torch.randn_like(xt)

    return SchedulerOutput(xt_next, P_xt)

def get_stable_diffusion_model(args):
    # load from hf
    model = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=args.dtype, cache_dir = args.cache_folder)

    model = model.to(args.device)

    model.enable_xformers_memory_efficient_attention()

    # turn-off xformers memory efficient attention for using forward AD
    # model.disable_xformers_memory_efficient_attention()
    
    # change scheduler
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)

    # change config (TODO need strict verification)
    model.vae.config.sample_size = 512

    # assert config
    assert model.scheduler.config.prediction_type == 'epsilon'
    return model


def get_latent_consistency_model(args):
    # load from hf
    model = DiffusionPipeline.from_pretrained(args.model_name, torch_dtype=args.dtype, cache_dir = args.cache_folder)
    model = model.to(args.device)
    # turn-off xformers memory efficient attention for using forward AD
    model.enable_xformers_memory_efficient_attention()

    # assert config
    assert model.scheduler.config.prediction_type == 'epsilon'
    return model

def get_latent_consistency_scheduler(args, scheduler):
    # monkey patch (replace scheduler for better inversion quality)
    if args.use_yh_custom_scheduler:
        scheduler.t_max = 999
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=args.device, dtype=args.dtype)
        scheduler.betas = scheduler.betas.to(device=args.device, dtype=args.dtype)
        scheduler.set_timesteps = types.MethodType(set_timesteps, scheduler)
        scheduler.step = types.MethodType(step, scheduler)
    else:
        pass
    return scheduler

def get_DeepFloyd_IF_model(args):
    # load from hf
    stage_1 = DiffusionPipeline.from_pretrained(args.model_name, torch_dtype=args.dtype, cache_dir = args.cache_folder)
    stage_1 = stage_1.to(torch_device = args.device)
    # stage_1.enable_model_cpu_offload()

    stage_1.unet.to(args.device)
    stage_1.text_encoder.to(args.device)
    stage_1.enable_xformers_memory_efficient_attention()

    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16, cache_dir = args.cache_folder
    )
    stage_2.enable_model_cpu_offload()
    stage_2.enable_xformers_memory_efficient_attention()

    safety_modules = {
        "feature_extractor": stage_1.feature_extractor,
        "safety_checker": stage_1.safety_checker,
        # "watermarker": stage_1.watermarker,
    }
    stage_3 = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
    )
    stage_3.enable_model_cpu_offload()
    stage_3.enable_xformers_memory_efficient_attention()
    
    # change scheduler

    stage_1.scheduler = DDIMScheduler.from_config(stage_1.scheduler.config)

    # assert config
    assert stage_1.scheduler.config.prediction_type == 'epsilon'
    return stage_1, stage_2, stage_3



###################
# diffusion utils #
###################
class SchedulerOutput(object):
    def __init__(self, xt_next, P_xt):
        self.prev_sample = xt_next
        self.x0 = P_xt
    
class YHCustomScheduler(object):
    def __init__(self, args):
        # NOTE : verify this
        self.t_max = 999
        self.noise_schedule = 'linear' if args.noise_schedule is None else args.noise_schedule
        self.timesteps = None
        self.learn_sigma = False

        # get SNR schedule
        self.get_alphas_cumprod(args)

    def set_timesteps(self, num_inferences, device=None, is_inversion=False):
        device = 'cpu' if device is None else device
        if is_inversion:
            seq = torch.linspace(0, 1, num_inferences, device=device) * self.t_max
            seq = seq + 1e-6
            seq_prev = torch.cat([torch.tensor([-1], device=device), seq[:-1]], dim = 0)
            self.timesteps = seq_prev[1:]
            self.timesteps_next = seq[1:]
        
        else:
            seq = torch.linspace(0, 1, num_inferences, device=device) * self.t_max
            seq_prev = torch.cat([torch.tensor([-1], device=device), seq[:-1]], dim = 0)
            self.timesteps = reversed(seq[1:])
            self.timesteps_next = reversed(seq_prev[1:])

    def get_timesteps(self, t):
        t_idx = torch.where(self.timesteps == t)
        # print("++++++++++++")
        # print(aa)
        # t_idx = self.timesteps.tolist().index(t)
        t_next  = self.timesteps_next[t_idx]
        return t_next

    def return_alphas_cumprod(self):
        return self.alphas_cumprod

    def step(self, et, t, xt, eta=0.0, **kwargs):
        '''
        Notation
            - a : alpha / b : beta / e : epsilon
        '''
        if self.learn_sigma:
            et, logvar = torch.split(et, et.shape[1] // 2, dim=1)
        else:
            logvar = None
        assert et.shape == xt.shape, 'et, xt shape should be same'

        t_idx   = self.timesteps.tolist().index(t)

        t_next  = self.timesteps_next[t_idx]
        
        # extract need parameters : at, at_next
        at = extract(self.alphas_cumprod, t, xt.shape)
        at_next = extract(self.alphas_cumprod, t_next, xt.shape)

        # DDIM step ; xt-1 = sqrt(at-1 / at) (xt - sqrt(1-at)*e(xt, t)) + sqrt(1-at-1)*e(xt, t)
        P_xt = (xt - et * (1 - at).sqrt()) / at.sqrt()

        # Deterministic.
        if eta == 0:
            D_xt = (1 - at_next).sqrt() * et
            xt_next = at_next.sqrt() * P_xt + D_xt

        # Add noise. When eta is 1 and time step is 1000, it is equal to ddpm.
        elif logvar is None:
            sigma_t = ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()

            D_xt = (1 - at_next - eta * sigma_t ** 2).sqrt() * et
            xt_next = at_next.sqrt() * P_xt + D_xt + eta * sigma_t * torch.randn_like(xt)

        elif logvar is not None:
            bt = extract(self.betas, t, xt.shape)
            
            mean = 1 / torch.sqrt(1.0 - bt) * (xt - bt / torch.sqrt(1 - at) * et)
            xt_next = mean + torch.exp(0.5 * logvar) * torch.randn_like(xt, device=xt.device, dtype=xt.dtype)
            P_xt = None

        return SchedulerOutput(xt_next, P_xt)

    def get_alphas_cumprod(self, args):
        # betas
        if self.noise_schedule == 'linear':
            betas = self.linear_beta_schedule(
                beta_start = 0.0001, # args.config.diffusion.beta_start,
                beta_end   = 0.02,   # args.config.diffusion.beta_end,
                timesteps  = 1000,   # args.config.diffusion.num_diffusion_timesteps,
            )

        elif self.noise_schedule == 'cosine':
            betas = self.cosine_beta_schedule(
                timesteps = self.t_max + 1
            )
        self.betas = betas.to(device=args.device, dtype=args.dtype)

        # alphas
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod = alphas_cumprod.to(device=args.device, dtype=args.dtype)

    def linear_beta_schedule(self, beta_start, beta_end, timesteps):
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
    
    # def cosine_beta_schedule(self, timesteps):
    #     return betas_for_alpha_bar(
    #         timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    #     )
    
    def cosine_beta_schedule(self, timesteps, s = 0.008):
        """
        cosine schedule (improved DDPM)
        proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                    produces the cumulative product of (1-beta) up to that
                    part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                    prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    # print(t.device)
    if isinstance(t, int):
        t = torch.tensor([t])
        t = t.repeat(x_shape[0])
    elif isinstance(t, torch.Tensor):
        t = t.repeat(x_shape[0])
    else:
        raise ValueError(f"t must be int or torch.Tensor, got {type(t)}")
    bs, = t.shape
    assert x_shape[0] == bs, f"{x_shape[0]}, {t.shape}"
    # print(a.device, t.device)
    out = torch.gather(a, 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out

###########
# dataset #
###########
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from dataset.celeba_hq_dataloader import CelebAMaskDataLoader

def get_dataset(args):
    '''
    Args
        - image_name : [Astronaut, Cyberpunk, VanGogh]
    Returns
        - dataset[idx] = img
    '''
    if args.dataset_name == 'Examples':
        dataset = ImgDataset(
            image_root = DATASET_PATHS['Examples'], 
            device = args.device, 
            dtype = args.dtype, 
            image_size = 512, # Stable-Diffusion
            dataset_name = 'Examples',
        )
    elif args.dataset_name == 'CelebA_HQ':
        dataset = ImgDataset(
            image_root = DATASET_PATHS['CelebA_HQ'], 
            device = args.device, 
            dtype = args.dtype, 
            image_size = 256, # High resolution DM
            dataset_name = 'CelebA_HQ',
        )
    elif args.dataset_name == 'LSUN_church':
        dataset = HFDataset(
            device = args.device, 
            dtype = args.dtype, 
            image_size = 256, # High resolution DM
            dataset_name = 'LSUN_church',
            dataset_id = "tglcourse/lsun_church_train")   
    elif args.dataset_name == 'LSUN_bedroom':
        dataset = HFDataset(
            device = args.device, 
            dtype = args.dtype, 
            image_size = 256, # High resolution DM
            dataset_name = 'LSUN_bedroom',
            dataset_id = "pcuenq/lsun-bedrooms")   
    elif args.dataset_name == 'CelebA_HQ_mask':          
        dataset = CelebAMaskDataLoader(
                root = args.dataset_root,
                save_path = os.path.join(args.result_folder, "dataset"))
    elif args.dataset_name == 'FFHQ':
        dataset = ImgDataset(
            image_root = args.dataset_root,
            device = args.device, 
            dtype = args.dtype, 
            image_size = 256, # High resolution DM
            dataset_name = 'FFHQ',
        )
    elif args.dataset_name == 'AFHQ':
        dataset = AFHQDataset(
            image_root = args.dataset_root,
            device = args.device, 
            dtype = args.dtype, 
            image_size = 256, # High resolution DM
            dataset_name = 'AFHQ',
        )
    elif args.dataset_name == 'Metface':
        dataset = HFDataset(
            device = args.device, 
            dtype = args.dtype, 
            image_size = 256, # High resolution DM
            dataset_name = 'Metface',
            dataset_id = "huggan/metfaces")
    elif args.dataset_name == 'Flower':
        dataset = HFDataset(
            device = args.device, 
            dtype = args.dtype, 
            image_size = 256, # High resolution DM
            dataset_name = 'Flower',
            dataset_id = "huggan/flowers-102-categories")        
    elif args.dataset_name == 'Random':
        dataset = None
    else:
        raise ValueError('Invalid dataset name')
    return dataset

class HFDataset(Dataset):
    def __init__(self, device, dtype, image_size, dataset_name, dataset_id):
        super().__init__()
        from datasets import load_dataset
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_id, split='train')
        self.transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])
        self.img_size = image_size
        self.device = device
        self.dtype = dtype

    def __getitem__(self, index):
        x = self.dataset['image'][index]
        
        # Crop the center of the image
        w, h = x.size
        crop_size = min(w, h)

        left    = (w - crop_size)/2
        top     = (h - crop_size)/2
        right   = (w + crop_size)/2
        bottom  = (h + crop_size)/2

        # Crop the center of the image
        x = x.crop((left, top, right, bottom))

        # resize the image
        x = x.resize((self.img_size, self.img_size))
        if self.transform is not None:
            x = self.transform(x) 
        return x.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def __len__(self):
        return len(self.image_paths)



class ImgDataset(Dataset):
    def __init__(self, image_root, device, dtype, image_size, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.image_dir = os.path.join(image_root)
        imgs_path_list = os.listdir(self.image_dir)
        imgs_path_list = [path for path in imgs_path_list if path.split('.')[1] in ['jpg', 'jpeg', 'png']]

        self.image_paths = sorted(imgs_path_list, key=lambda path: int(path.split('.')[0]))
        self.transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])
        self.img_size = image_size
        self.device = device
        self.dtype = dtype

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        x = Image.open(image_path)
        
        # Crop the center of the image
        w, h = x.size
        crop_size = min(w, h)

        left    = (w - crop_size)/2
        top     = (h - crop_size)/2
        right   = (w + crop_size)/2
        bottom  = (h + crop_size)/2

        # Crop the center of the image
        x = x.crop((left, top, right, bottom))

        # resize the image
        x = x.resize((self.img_size, self.img_size))
        if self.transform is not None:
            x = self.transform(x) 
        return x.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def __len__(self):
        return len(self.image_paths)

class AFHQDataset(Dataset):
    def __init__(self, image_root, device, dtype, image_size, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.image_dir = os.path.join(image_root)
        imgs_path_list = os.listdir(self.image_dir)
        imgs_path_list = [path for path in imgs_path_list if path.split('.')[1] in ['jpg', 'jpeg', 'png']]

        self.image_paths = sorted(imgs_path_list, key=lambda path: (path.split('/')[-1].split('.')[0]))
        self.transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])
        self.img_size = image_size
        self.device = device
        self.dtype = dtype

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        print(image_path)
        x = Image.open(image_path)
        
        # Crop the center of the image
        w, h = x.size
        crop_size = min(w, h)

        left    = (w - crop_size)/2
        top     = (h - crop_size)/2
        right   = (w + crop_size)/2
        bottom  = (h + crop_size)/2

        # Crop the center of the image
        x = x.crop((left, top, right, bottom))

        # resize the image
        x = x.resize((self.img_size, self.img_size))
        if self.transform is not None:
            x = self.transform(x) 
        return x.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def __len__(self):
        return len(self.image_paths)

