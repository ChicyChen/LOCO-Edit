import gc
import os
from tqdm import tqdm
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as tvu

import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, einsum
from diffusers.utils import pt_to_pil
from utils.utils import (
    get_dataset,
    get_stable_diffusion_model,
    get_DeepFloyd_IF_model,
    get_custom_diffusion_model,
    get_custom_diffusion_scheduler,
    get_stable_diffusion_scheduler,
    get_deepfloyd_if_scheduler,
    extract,
    concatenate_pil_horizontally,
    get_prependicualr_component,
    get_latent_consistency_model,
    get_latent_consistency_scheduler,
)
from copy import deepcopy
######################
# LDM ; use diffuser #
######################
from diffusers import (
    # DDIMInverseScheduler,
    DDIMScheduler, 
)

from modules.mask_segmentation import SAM


class EditLatentConsistency(object):
    def __init__(self, args):
        # default setting
        self.seed = args.seed
        self.pca_device     = args.pca_device
        self.buffer_device  = args.buffer_device
        self.memory_bound   = args.memory_bound

        # path
        self.result_folder = os.path.join(args.result_folder, f"for_prompt_{args.for_prompt}_cfg{args.guidance_scale}_seed{args.seed}")
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.obs_folder = args.obs_folder

        # get model
        self.pipe = get_latent_consistency_model(args)
        self.vae  = self.pipe.vae
        self.unet = self.pipe.unet
        self.sam = SAM(args, log_dir = self.result_folder)

        self.dtype  = args.dtype
        self.device = self.pipe._execution_device

        # args (diffusion schedule)
        self.scheduler = get_latent_consistency_scheduler(args, self.pipe.scheduler)
        # self.for_steps = args.for_steps
        # self.inv_steps = args.inv_steps
        self.use_yh_custom_scheduler = args.use_yh_custom_scheduler

        # get image
        self.c_in = args.c_in
        self.image_size = args.image_size
        self.dataset = get_dataset(args)
        self.dataset_name = args.dataset_name

        # args (prompt)
        self.for_prompt         = args.for_prompt 
        self.edit_prompt         = args.edit_prompt 
        
        # args (guidance)
        self.guidance_scale     = args.guidance_scale
        self.guidance_scale_edit= args.guidance_scale_edit

        # x-space guidance
        self.x_edit_step_size = args.x_edit_step_size
        self.x_space_guidance_edit_step         = args.x_space_guidance_edit_step
        self.x_space_guidance_scale             = args.x_space_guidance_scale
        self.x_space_guidance_num_step          = args.x_space_guidance_num_step
        self.x_space_guidance_use_edit_prompt   = args.x_space_guidance_use_edit_prompt


        self.num_inference_steps = args.num_inference_steps
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        
        self.edit_t         = args.edit_t
        self.edit_t_idx     = args.edit_t_idx

        self.sampling_mode  = args.sampling_mode
        self.use_sega = args.use_sega

    @torch.no_grad()
    def run_LCMforward(self, zT, prompt, num_samples=1):
        print('start LCMforward')

        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = False,
        )

        # get latent code
        # zT = torch.randn(num_samples, 4, 64, 64).to(device=self.device, dtype=self.dtype)
        latents = zT
        w = torch.tensor(self.guidance_scale - 1).repeat(1)
        w_embedding = self.pipe.get_guidance_scale_embedding(w, embedding_dim=self.unet.config.time_cond_proj_dim).to(
            device=self.device, dtype=latents.dtype
        )

        for t_idx, t in enumerate(self.scheduler.timesteps):
            latents = latents.to(device=self.device)

            model_pred = self.unet(
                    latents,
                    t,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents, denoised = self.scheduler.step(model_pred, t, latents, return_dict=False)

        denoised = denoised.to(prompt_embeds.dtype)
        image = self.vae.decode(denoised / self.vae.config.scaling_factor, return_dict=False)[0]

        x0 = (image / 2 + 0.5).clamp(0, 1)
        tvu.save_image(x0, os.path.join(self.result_folder, f'{self.EXP_NAME}.png'), nrow = x0.size(0))
        x0 = (x0 * 255).to(torch.uint8).permute(0, 2, 3, 1)

        return latents, x0


    @torch.no_grad()
    def LCMforwardsteps(self, zt, prompt, t_start_idx=0, t_end_idx=-1):
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        latents = zt

        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = False,
        )
        prompt_embeds = prompt_embeds.repeat(latents.shape[0],1,1)
        
        w = torch.tensor(self.guidance_scale - 1).repeat(1)
        w_embedding = self.pipe.get_guidance_scale_embedding(w, embedding_dim=self.unet.config.time_cond_proj_dim).to(
            device=self.device, dtype=latents.dtype
        )
        w_embedding = w_embedding.repeat(latents.shape[0],1)

        for t_idx, t in enumerate(self.scheduler.timesteps):
            # skip
            if (t_idx < t_start_idx): 
                continue

            # start sampling
            elif t_start_idx == t_idx:
                # print('t_start_idx : ', t_idx)
                pass

            # end sampling
            elif t_idx == t_end_idx:
                # print('t_end_idx : ', t_idx)
                return latents, t, t_idx

            # print("t_idx", t_idx)
            latents = latents.to(device=self.device)

            model_pred = self.unet(
                    latents,
                    t,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents, denoised = self.scheduler.step(model_pred, t, latents, return_dict=False)

        # denoised = denoised.to(prompt_embeds.dtype)
        image = self.vae.decode(denoised / self.vae.config.scaling_factor, return_dict=False)[0]

        x0 = (image / 2 + 0.5).clamp(0, 1)
        tvu.save_image(x0, os.path.join(self.result_folder, f'{self.EXP_NAME}.png'), nrow = x0.size(0))
        x0 = (x0 * 255).to(torch.uint8).permute(0, 2, 3, 1)

        return latents, x0


    def get_x0(self, zt, prompt, t, t_idx, mask=None, flatten=False):
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = False,
        )

        latents = zt
        w = torch.tensor(self.guidance_scale - 1).repeat(1)
        w_embedding = self.pipe.get_guidance_scale_embedding(w, embedding_dim=self.unet.config.time_cond_proj_dim).to(
            device=self.device, dtype=latents.dtype
        )
        latents = latents.to(device=self.device)

        # broadcast
        prompt_embeds = prompt_embeds.repeat(latents.shape[0], 1, 1)
        w_embedding = w_embedding.repeat(latents.shape[0], 1)

        model_pred = self.unet(
                latents,
                t,
                timestep_cond=w_embedding,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

        _, z0_hat = self.scheduler.step(model_pred, t, latents, return_dict=False)
        x0_hat = self.vae.decode(z0_hat / self.vae.config.scaling_factor, return_dict=False)[0]

        if mask is not None:
            x0_hat = x0_hat[:, mask]
            return x0_hat

        # only use this for single xt and no mask:
        if flatten:
            c_i, w_i, h_i = x0_hat.size(1), x0_hat.size(2), x0_hat.size(3)
            x0_hat = x0_hat.view(-1, c_i*w_i*h_i)

        return x0_hat


    @torch.no_grad()
    def get_delta_zt_via_grad(self, zt, t, t_idx, for_prompt, edit_prompt, mask = None):
        x0_hat = self.get_x0(zt, for_prompt, t, t_idx)
        x0_hat_after = self.get_x0(zt, edit_prompt, t, t_idx)
        x0_hat_delta = x0_hat_after - x0_hat

        # tvu.save_image((x0_hat / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'x0_hat_for.png'))
        # tvu.save_image((x0_hat_after / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'x0_hat_edit.png'))
        # tvu.save_image((x0_hat_delta / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'x0_hat_delta.png'))

        c_i, w_i, h_i = zt.size(1), zt.size(2), zt.size(3)
        c_o, w_o, h_o = x0_hat.size(1), x0_hat.size(2), x0_hat.size(3)

        # we can even add mask here
        if mask is not None:
            x0_hat_delta_flat = x0_hat_delta[:,mask]
            # print(x0_hat_delta_flat.shape)
        else:
            x0_hat_delta_flat = x0_hat_delta.view(-1,c_o*w_o*h_o)

        g = lambda v : torch.sum(x0_hat_delta_flat * self.get_x0(v, edit_prompt, t, t_idx, mask = mask, flatten=True))
        v_ = torch.autograd.functional.jacobian(g, zt)
        v_ = v_.view(-1, c_i*w_i*h_i)
        v_ = v_ / v_.norm(dim=1, keepdim=True)

        zt_delta = v_.view(-1, c_i, w_i, h_i)
        zt_new = zt + 20.0 * zt_delta
        x0_hat_new = self.get_x0(zt_new, for_prompt, t, t_idx)
        # tvu.save_image((x0_hat_new / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'x0_hat_viagrad.png'))
        
        return v_


    def local_encoder_decoder_pullback_zt(
            self, zt, t, t_idx, for_prompt, op=None, block_idx=None,
            pca_rank=50, chunk_size=25, min_iter=10, max_iter=100, convergence_threshold=1e-3, mask = None
        ):
        '''
        Args
            - zt : zt
            - op : ['down', 'mid', 'up']
            - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
            - pooling : ['pixel-sum', 'channel-sum', 'single-channel', 'multiple-channel']
        Returns
            - h : hidden feature
        '''
        # assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)"]
        num_chunk = pca_rank // chunk_size if pca_rank % chunk_size == 0 else pca_rank // chunk_size + 1

        # get h samples
        time_s = time.time()

        c_i, w_i, h_i = zt.size(1), zt.size(2), zt.size(3)
        if mask is None:
            c_o, w_o, h_o = c_i, w_i, h_i # output shape of x^0
        else:
            l_o = mask.sum().item()


        a = torch.tensor(0., device=zt.device, dtype=zt.dtype)

        # Algorithm 1
        vT = torch.randn(c_i*w_i*h_i, pca_rank, device=zt.device, dtype=torch.float)
        vT, _ = torch.linalg.qr(vT)
        v = vT.T
        v = v.view(-1, c_i, w_i, h_i)


        time_s = time.time()
        # Jacobian subspace iteration
        for i in range(max_iter):
            v = v.to(device=zt.device, dtype=zt.dtype)
            v_prev = v.detach().cpu().clone()
            
            u = []
            v_buffer = list(v.chunk(num_chunk))
            for vi in v_buffer:
                # print((zt + a*vi).shape)
                g = lambda a : self.get_x0(zt + a*vi, for_prompt, t, t_idx, mask = mask)
                
                ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='same')(a) # ui = J@vi
                u.append(ui.detach().cpu().clone())
            u = torch.cat(u, dim=0)
            u = u.to(zt.device, zt.dtype)

            if mask is None:
                g = lambda zt : einsum(
                    u, self.get_x0(zt, for_prompt, t, t_idx, mask=mask), 'b c w h, i c w h -> b'
                )
            else:
                g = lambda zt : einsum(
                    u, self.get_x0(zt, for_prompt, t, t_idx, mask=mask), 'b l, i l -> b'
                )                
            
            v_ = torch.autograd.functional.jacobian(g, zt) # vi = ui.T@J
            v_ = v_.view(-1, c_i*w_i*h_i)

            _, s, v = torch.linalg.svd(v_, full_matrices=False)
            v = v.view(-1, c_i, w_i, h_i)
            if mask is None:
                u = u.view(-1, c_o, w_o, h_o)
            else:
                u = u.view(-1, l_o)
            
            convergence = torch.dist(v_prev, v.detach().cpu()).item()
            print(f'power method : {i}-th step convergence : ', convergence)
            
            if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
                print('reach convergence threshold : ', convergence)
                break

        time_e = time.time()
        print('power method runtime ==', time_e - time_s)

        if mask is None:
            u, s, vT = u.reshape(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()
        else:
            u, s, vT = u.reshape(-1, l_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()

        return u, s, vT



    @torch.no_grad()
    def run_edit_null_space_projection_zt(
            self, op, block_idx, vis_num, mask_index = 0, vis_num_pc=1, vis_vT=False, pca_rank=50, edit_prompt=None, null_space_projection = False, pca_rank_null=50, 
            non_semantic = False
        ):
        # set edit prompt
        if edit_prompt is not None:
            self.edit_prompt = edit_prompt

        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        # get latent code (zT -> zt)
        if self.dataset_name == 'Random':
            zT = torch.randn(1, 4, 64, 64, dtype=self.dtype).to(self.device)

        self.EXP_NAME = "original"
        if (not os.path.exists(os.path.join(self.result_folder, "original.png"))) or (not os.path.exists(os.path.join(self.result_folder, "mask/mask.pt"))):
            print("Generating images and creating masks......")
            _, x0 = self.run_LCMforward(zT, prompt=self.for_prompt)
            masks = self.sam.mask_segmentation(Image.fromarray(np.array(x0[0].detach().cpu())), resolution=512)
        else:
            print("Loading masks......")
            masks = torch.load(os.path.join(self.result_folder, "mask/mask.pt"))
        
        if self.sampling_mode:
            return None
        mask = masks[mask_index].squeeze(dim=0).repeat(3, 1, 1)

        zt, t, t_idx = self.LCMforwardsteps(zT, t_start_idx=0, t_end_idx=self.edit_t_idx, 
                                            prompt=self.for_prompt)
        assert t_idx == self.edit_t_idx
        
        # get local basis
        if not self.use_sega:
            print('!!!RUN LOCAL PULLBACK!!!')

            if non_semantic:
                _, _, vT_modify = self.local_encoder_decoder_pullback_zt(
                zt, t, t_idx, self.for_prompt, op=op, block_idx=block_idx,
                pca_rank=pca_rank, chunk_size=5, min_iter=10, max_iter=50, 
                convergence_threshold=1e-3, mask = mask
                )
            else:
                vT_modify = self.get_delta_zt_via_grad(zt.clone(), t, t_idx, self.for_prompt, self.edit_prompt, mask = mask)


            if null_space_projection:
                _, _, vT_null = self.local_encoder_decoder_pullback_zt(
                zt, t, t_idx, self.for_prompt, op=op, block_idx=block_idx,
                pca_rank=pca_rank_null, chunk_size=5, min_iter=10, max_iter=50, 
                convergence_threshold=1e-3, mask = ~mask
                )

            # normalize u, vT
            if not null_space_projection:
                vT = vT_modify
            else:
                vT_null = vT_null[:pca_rank_null, :]
                vT = (vT_null.T @ (vT_null @ vT_modify.T)).T
                vT = vT_modify - vT
            vT = vT / vT.norm(dim=1, keepdim=True)


            original_zt = zt.clone()

            zts = {
                -1: None,
                1: None,
            }
            self.EXP_NAME = f'Edit_zt-edit_{self.edit_t_idx}T-{op}-block_{block_idx}_pos-edit_prompt-{self.edit_prompt}_select_mask{mask_index}_null_space_projection_{null_space_projection}_null_space_rank_{pca_rank_null}'
            
            for direction in [1, -1]:
                vk = direction * vT.view(-1, *zT.shape[1:])
                # edit zt along vk direction with **x-space guidance**
                zt_list = [original_zt.clone()]
                for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                    zt_edit = self.x_space_guidance_direct(
                        zt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                        single_edit_step=self.x_space_guidance_edit_step
                    )
                    zt_list.append(zt_edit)
                zt = torch.cat(zt_list, dim=0)
                if vis_num == 1:
                    zt = zt[[0,-1],:]
                else:
                    zt = zt[::(zt.size(0) // vis_num)]
                # zt = zt[::(zt.size(0) // vis_num)]
                zts[direction] = zt
                # zt -> z0
            zt = torch.cat([(zts[-1].flip(dims=[0]))[:-1], zts[1]], dim=0)

            self.LCMforwardsteps(
            zt, t_start_idx=self.edit_t_idx, t_end_idx=-1, prompt=self.for_prompt)


        else:
            self.EXP_NAME = f'sega_{self.edit_t_idx}T-{op}-block_{block_idx}_pos-edit_prompt-{self.edit_prompt}'
            self.LCMforwardsteps(
                zt, t_start_idx=self.edit_t_idx, t_end_idx=-1, prompt=self.edit_prompt)
        

    @torch.no_grad()
    def x_space_guidance_direct(self, zt, t_idx, vk, single_edit_step):
        # edit xt with vk
        zt_edit = zt + self.x_space_guidance_scale * single_edit_step * vk

        return zt_edit



class EditStableDiffusion(object):
    def __init__(self, args):
        # default setting
        self.seed = args.seed
        self.pca_device     = args.pca_device
        self.buffer_device  = args.buffer_device
        self.memory_bound   = args.memory_bound

        # path
        self.result_folder = os.path.join(args.result_folder, f"for_prompt_{args.for_prompt}_cfg{args.guidance_scale}_seed{args.seed}")
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.obs_folder = args.obs_folder

        # get model
        self.pipe = get_stable_diffusion_model(args)
        self.vae  = self.pipe.vae
        self.unet = self.pipe.unet
        self.sam = SAM(args, log_dir = self.result_folder)

        self.dtype  = args.dtype
        self.device = self.pipe._execution_device

        # args (diffusion schedule)
        self.scheduler = get_stable_diffusion_scheduler(args, self.pipe.scheduler)
        self.for_steps = args.for_steps
        self.inv_steps = args.inv_steps
        self.use_yh_custom_scheduler = args.use_yh_custom_scheduler

        # get image
        self.c_in = args.c_in
        self.image_size = args.image_size
        self.dataset = get_dataset(args)
        self.dataset_name = args.dataset_name

        # args (prompt)
        self.for_prompt         = args.for_prompt 
        #if len(args.for_prompt.split(',')[0]) <= 3 else ','.join([args.for_prompt.split(',')[0]])
        self.neg_prompt         = args.neg_prompt 
        #if len(args.neg_prompt.split(',')[0]) <= 3 else ','.join([args.neg_prompt.split(',')[0]])
        self.null_prompt        = ""
        self.inv_prompt         = args.inv_prompt if len(args.inv_prompt.split(',')[0]) <= 3 else ','.join([args.inv_prompt.split(',')[0]])

        self.for_prompt_emb     = self._get_prompt_emb(args.for_prompt)
        self.neg_prompt_emb     = self._get_prompt_emb(args.neg_prompt)
        self.null_prompt_emb    = self._get_prompt_emb("")
        self.inv_prompt_emb     = self._get_prompt_emb(args.inv_prompt)
        
        # args (guidance)
        self.guidance_scale     = args.guidance_scale
        self.guidance_scale_edit= args.guidance_scale_edit
        

        # args (h space edit)        
        self.edit_prompt        = args.edit_prompt 
        self.edit_prompt_emb    = self._get_prompt_emb(args.edit_prompt)

        # x-space guidance
        self.x_edit_step_size = args.x_edit_step_size
        self.x_space_guidance_edit_step         = args.x_space_guidance_edit_step
        self.x_space_guidance_scale             = args.x_space_guidance_scale
        self.x_space_guidance_num_step          = args.x_space_guidance_num_step
        self.x_space_guidance_use_edit_prompt   = args.x_space_guidance_use_edit_prompt


        self.scheduler.set_timesteps(self.for_steps, device=self.device)
        self.edit_t         = args.edit_t
        self.edit_t_idx     = (self.scheduler.timesteps - self.edit_t * 1000).abs().argmin()
        self.sampling_mode  = args.sampling_mode
        self.use_sega = args.use_sega
        self.tilda_v_score_type = args.tilda_v_score_type


    @torch.no_grad()
    def run_DDIMforward(self, num_samples=5):
        print('start DDIMforward')
        self.EXP_NAME = f'DDIMforward-for_{self.for_prompt}'

        # get latent code
        zT = torch.randn(num_samples, 4, 64, 64).to(device=self.device, dtype=self.dtype)

        # simple DDIMforward
        self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=-1)

    @torch.no_grad()
    def run_DDIMinversion(self, idx, guidance=None, vis_traj=False):
        '''
        Prompt
            (CFG)       pos : inv_prompt, neg : null_prompt
            (no CFG)    pos : inv_prompt
        '''
        print('start DDIMinversion')
        self.EXP_NAME = f'DDIMinversion-{self.dataset_name}-{idx}-for_{self.for_prompt}-inv_{self.inv_prompt}'

        # inversion scheduler

        # before start
        num_inference_steps = self.inv_steps
        do_classifier_free_guidance = (self.guidance_scale > 1.0) & (guidance is not None)

        # set timestep (we do not use default scheduler set timestep method)
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, is_inversion=True)
        else:
            raise ValueError('recommend to use yh custom scheduler')
            self.scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # get image
        x0 = self.dataset[idx]
        tvu.save_image((x0 / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'original_x0.png'))

        # get latent
        z0 = self.vae.encode(x0).latent_dist
        z0 = z0.sample()
        z0 = z0 * 0.18215

        ##################
        # denoising loop #
        ##################
        latents = z0
        for i, t in enumerate(timesteps):
            if i == len(timesteps) - 1:
                break

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if do_classifier_free_guidance:
                prompt_emb = torch.cat([self.null_prompt_emb.repeat(latents.size(0), 1, 1), self.inv_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)
            else:
                prompt_emb = self.inv_prompt_emb.repeat(latents.size(0), 1, 1)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=prompt_emb,
                # cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, eta=0).prev_sample

        return latents


    def _classifer_free_guidance(self, latents, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode, do_classifier_free_guidance):
        assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)"]
        if do_classifier_free_guidance:
            if mode == "null+(for-null)":
                latent_model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)
            elif mode == "null+(for-null)+(edit-null)":
                latent_model_input = torch.cat([latents] * 3, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), edit_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)
            elif mode == "null+(edit-null)":
                latent_model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([edit_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)    
            elif mode == "(for-edit)":
                latent_model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), edit_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)                               
        else:
            latent_model_input = latents
            prompt_emb = for_prompt_emb.repeat(latents.size(0), 1, 1)
    
        noise_pred = self.unet(
            latent_model_input, t,
            encoder_hidden_states=prompt_emb,
        ).sample

        # perform guidance
        if do_classifier_free_guidance:
            if mode == "null+(for-null)+(edit-null)":
                noise_pred_for, noise_pred_edit, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_for - noise_pred_uncond) + self.guidance_scale_edit * (noise_pred_edit - noise_pred_uncond)
            elif mode == "null+(for-null)":
                noise_pred_for, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_for - noise_pred_uncond)
            elif mode == "null+(edit-null)":
                noise_pred_edit, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_edit - noise_pred_uncond)
            elif mode == "(for-edit)":
                noise_pred_for, noise_pred_edit = noise_pred.chunk(2)
                noise_pred = self.guidance_scale * (noise_pred_for - noise_pred_edit)                        
        return noise_pred

    @torch.no_grad()
    def DDIMforwardsteps(
            self, zt, t_start_idx, t_end_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = "null+(for-null)", **kwargs
        ):
        '''
        Prompt
            (CFG)       pos : for_prompt, neg : neg_prompt
            (no CFG)    pos : for_prompt
        '''
        assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","edit-proj[for](edit)","null+for+edit-proj[for](edit)"]
        print('start DDIMforward')
        # before start
        num_inference_steps = self.for_steps
        do_classifier_free_guidance = self.guidance_scale > 1.0
        # cross_attention_kwargs      = None
        memory_bound = self.memory_bound // 2 if do_classifier_free_guidance else self.memory_bound
        print(memory_bound)
        print('do_classifier_free_guidance : ', do_classifier_free_guidance)

        # set timestep (we do not use default scheduler set timestep method)
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        else:
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # save traj
        latents = zt

        #############################################
        # denoising loop (t_start_idx -> t_end_idx) #
        #############################################
        for t_idx, t in enumerate(self.scheduler.timesteps):
            # skip
            if (t_idx < t_start_idx): 
                continue
                
            # start sampling
            elif t_start_idx == t_idx:
                # print('t_start_idx : ', t_idx)
                pass

            # end sampling
            elif t_idx == t_end_idx:
                # print('t_end_idx : ', t_idx)
                return latents, t, t_idx

            # split zt to avoid OOM
            latents = latents.to(device=self.buffer_device)
            if latents.size(0) == 1:
                latents_buffer = [latents]
            else:
                latents_buffer = list(latents.chunk(latents.size(0) // memory_bound))

            # loop over buffer
            for buffer_idx, latents in enumerate(latents_buffer):
                # overload to device
                latents = latents.to(device=self.device)

                noise_pred = self._classifer_free_guidance(latents, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = mode, do_classifier_free_guidance = do_classifier_free_guidance)
                # print("device check:", t.device)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, eta=0).prev_sample

                # save latents in buffer
                latents_buffer[buffer_idx] = latents.to(self.buffer_device)

            latents = torch.cat(latents_buffer, dim=0)
            latents = latents.to(device=self.device)
            del latents_buffer
            torch.cuda.empty_cache()

        # decode with vae
        latents = 1 / 0.18215 * latents
        x0 = self.vae.decode(latents).sample
        x0 = (x0 / 2 + 0.5).clamp(0, 1)
        tvu.save_image(x0, os.path.join(self.result_folder, f'{self.EXP_NAME}.png'), nrow = x0.size(0))
        x0 = (x0 * 255).to(torch.uint8).permute(0, 2, 3, 1)
        return latents, x0


    def get_x0(self, zt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = None, mode = "null+(for-null)+(edit-null)", flatten=False):
        assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)"]
        
        do_classifier_free_guidance = self.guidance_scale > 1.0

        noise_pred = self._classifer_free_guidance(zt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = mode, do_classifier_free_guidance = do_classifier_free_guidance)

        at = extract(self.scheduler.alphas_cumprod, t, zt.shape)

        z0_hat = (zt - noise_pred * (1 - at).sqrt()) / at.sqrt()

        # decode
        z0_hat = 1 / 0.18215 * z0_hat
        x0_hat = self.vae.decode(z0_hat).sample

        if mask is not None:
            x0_hat = x0_hat[:, mask]
            return x0_hat

        # only use this for single xt and no mask:
        if flatten:
            # print(xt.shape, x0_hat.shape)
            c_i, w_i, h_i = x0_hat.size(1), x0_hat.size(2), x0_hat.size(3)
            x0_hat = x0_hat.view(-1, c_i*w_i*h_i)
        return x0_hat

    @torch.no_grad()
    def get_delta_zt_via_grad(self, zt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = None, mode = "null+(for-null)+(edit-null)"):
        # assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)"]
        
        do_classifier_free_guidance = self.guidance_scale > 1.0

        noise_pred = self._classifer_free_guidance(zt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = "null+(for-null)", do_classifier_free_guidance = do_classifier_free_guidance)
        at = extract(self.scheduler.alphas_cumprod, t, zt.shape)
        z0_hat = (zt - noise_pred * (1 - at).sqrt()) / at.sqrt()
        # decode
        z0_hat = 1 / 0.18215 * z0_hat
        x0_hat = self.vae.decode(z0_hat).sample

        noise_pred_after= self._classifer_free_guidance(zt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = mode, do_classifier_free_guidance = do_classifier_free_guidance)
        at = extract(self.scheduler.alphas_cumprod, t, zt.shape)
        z0_hat_after = (zt - noise_pred_after * (1 - at).sqrt()) / at.sqrt()
        # decode
        z0_hat_after = 1 / 0.18215 * z0_hat_after
        x0_hat_after = self.vae.decode(z0_hat_after).sample

        x0_hat_delta = x0_hat_after - x0_hat
        c_i, w_i, h_i = zt.size(1), zt.size(2), zt.size(3)

        # we can even add mask here
        if mask is not None:
            x0_hat_delta_flat = x0_hat_delta[:,mask]
            # print(x0_hat_delta_flat.shape)
        else:
            x0_hat_delta_flat = x0_hat_delta.view(-1,c_i*w_i*h_i)

        g = lambda v : torch.sum(x0_hat_delta_flat * self.get_x0(v, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = mask, mode = mode, flatten=True))
        v_ = torch.autograd.functional.jacobian(g, zt)

        v_ = v_.view(-1, c_i*w_i*h_i)
        v_ = v_ / v_.norm(dim=1, keepdim=True)

        zt_delta = v_.view(-1, c_i, w_i, h_i)
        zt_new = 10.0 * zt_delta + zt
        noise_pred_new = self._classifer_free_guidance(zt_new, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = "null+(for-null)", do_classifier_free_guidance = do_classifier_free_guidance)
        z0_hat_new = (zt_new - noise_pred_new * (1 - at).sqrt()) / at.sqrt()
        # decode
        z0_hat_new = 1 / 0.18215 * z0_hat_new
        x0_hat_new = self.vae.decode(z0_hat_new).sample
        # tvu.save_image((x0_hat_new / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'viagrad_x0_hat.png'))
        
        return v_

    def local_encoder_decoder_pullback_zt(
            self, zt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, op=None, block_idx=None,
            pca_rank=50, chunk_size=25, min_iter=10, max_iter=100, convergence_threshold=1e-3, mask = None,
            mode = "null+(for-null)+(edit-null)"
        ):
        '''
        Args
            - zt : zt
            - op : ['down', 'mid', 'up']
            - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
            - pooling : ['pixel-sum', 'channel-sum', 'single-channel', 'multiple-channel']
        Returns
            - h : hidden feature
        '''
        assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)","edit-proj[for](edit)","null+for+edit-proj[for](edit)"]
        num_chunk = pca_rank // chunk_size if pca_rank % chunk_size == 0 else pca_rank // chunk_size + 1

        # get h samples
        time_s = time.time()

        c_i, w_i, h_i = zt.size(1), zt.size(2), zt.size(3)
        if mask is None:
            c_o, w_o, h_o = c_i, w_i, h_i # output shape of x^0
        else:
            l_o = mask.sum().item()


        a = torch.tensor(0., device=zt.device, dtype=zt.dtype)

        # Algorithm 1
        vT = torch.randn(c_i*w_i*h_i, pca_rank, device=zt.device, dtype=torch.float)
        vT, _ = torch.linalg.qr(vT)
        v = vT.T
        v = v.view(-1, c_i, w_i, h_i)


        time_s = time.time()
        # Jacobian subspace iteration
        for i in range(max_iter):
            v = v.to(device=zt.device, dtype=zt.dtype)
            v_prev = v.detach().cpu().clone()
            
            u = []
            v_buffer = list(v.chunk(num_chunk))
            for vi in v_buffer:
                g = lambda a : self.get_x0(zt + a*vi, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = mask, mode=mode)
                
                ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a) # ui = J@vi
                u.append(ui.detach().cpu().clone())
            u = torch.cat(u, dim=0)
            u = u.to(zt.device, zt.dtype)

            if mask is None:
                g = lambda zt : einsum(
                    u, self.get_x0(zt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask=mask, mode=mode), 'b c w h, i c w h -> b'
                )
            else:
                g = lambda zt : einsum(
                    u, self.get_x0(zt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask=mask, mode=mode), 'b l, i l -> b'
                )                
            
            v_ = torch.autograd.functional.jacobian(g, zt) # vi = ui.T@J
            v_ = v_.view(-1, c_i*w_i*h_i)

            _, s, v = torch.linalg.svd(v_, full_matrices=False)
            v = v.view(-1, c_i, w_i, h_i)
            if mask is None:
                u = u.view(-1, c_o, w_o, h_o)
            else:
                u = u.view(-1, l_o)
            
            convergence = torch.dist(v_prev, v.detach().cpu()).item()
            print(f'power method : {i}-th step convergence : ', convergence)
            
            if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
                print('reach convergence threshold : ', convergence)
                break

        time_e = time.time()
        print('power method runtime ==', time_e - time_s)

        if mask is None:
            u, s, vT = u.reshape(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()
        else:
            u, s, vT = u.reshape(-1, l_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()
        return u, s, vT

    @torch.no_grad()
    def run_edit_null_space_projection_zt(
            self, op, block_idx, vis_num, mask_index = 0, vis_num_pc=1, vis_vT=False, pca_rank=50, edit_prompt=None, null_space_projection = False, pca_rank_null=50, 
            non_semantic = False
        ):
        print(f'current experiment : op : {op}, block_idx : {block_idx}, vis_num : {vis_num}, vis_num_pc : {vis_num_pc}, pca_rank : {pca_rank}, edit_prompt : {edit_prompt}, null_space_projection = {null_space_projection}, pca_rank_null={pca_rank_null}')
        '''
        1. z0 -> zT -> zt -> z0 ; we edit latent variable zt
        2. get local basis of h-space (u) and x-space (v) by using the power method
        3. edit sample with x-space guidance
        '''
        # set edit prompt
        if edit_prompt is not None:
            self.edit_prompt = edit_prompt
            self.edit_prompt_emb = self._get_prompt_emb(self.edit_prompt)

        # set edit_t
        self.scheduler.set_timesteps(self.for_steps)

        # get latent code (zT -> zt)
        if self.dataset_name == 'Random':
            zT = torch.randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)
        
        self.EXP_NAME = "original"
        if (not os.path.exists(os.path.join(self.result_folder, "original.png"))) or (not os.path.exists(os.path.join(self.result_folder, "mask/mask.pt"))):
            print("Generating images and creating masks......")
            _, x0 = self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=-1, 
                                          for_prompt_emb=self.for_prompt_emb, 
                                          edit_prompt_emb=self.edit_prompt_emb, 
                                          null_prompt_emb=self.null_prompt_emb,
                                          mode="null+(for-null)")
            masks = self.sam.mask_segmentation(Image.fromarray(np.array(x0[0].detach().cpu())), resolution=512)

        else:
            print("Loading masks......")
            masks = torch.load(os.path.join(self.result_folder, "mask/mask.pt"))
        
        if self.sampling_mode:
            return None
        mask = masks[mask_index].squeeze(dim=0).repeat(3, 1, 1)
        
        zt, t, t_idx = self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=self.edit_t_idx, 
                                            for_prompt_emb=self.for_prompt_emb, 
                                            edit_prompt_emb=self.edit_prompt_emb, 
                                            null_prompt_emb=self.null_prompt_emb,
                                            mode="null+(for-null)")
        assert t_idx == self.edit_t_idx

        # get local basis
        save_dir = os.path.join(self.result_folder, "basis", f'local_basis-{self.edit_t}T-pca-rank-{pca_rank}-select-mask{mask_index}')
        os.makedirs(save_dir, exist_ok=True)
        u_modify_path = os.path.join(save_dir, f'u-modify.pt')
        vT_modify_path = os.path.join(save_dir, f'vT-modify.pt')
        u_null_path = os.path.join(save_dir, f'u-null-null_space_rank_{pca_rank_null}.pt')
        vT_null_path = os.path.join(save_dir, f'vT-null-null_space_rank_{pca_rank_null}.pt')         
        # load pre-computed local basis
        if os.path.exists(u_modify_path) and os.path.exists(vT_modify_path) and os.path.exists(u_null_path) and os.path.exists(vT_null_path):
            u_modify = torch.load(u_modify_path, map_location=self.device).type(self.dtype)
            vT_modify = torch.load(vT_modify_path, map_location=self.device).type(self.dtype)
            u_null = torch.load(u_null_path, map_location=self.device).type(self.dtype)
            vT_null = torch.load(vT_null_path, map_location=self.device).type(self.dtype)

        else:
            print('!!!RUN LOCAL PULLBACK!!!')
            zt = zt.to(device=self.device, dtype=self.dtype)

            u_modify, s_modify, vT_modify = self.local_encoder_decoder_pullback_zt(
                zt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, op=op, block_idx=block_idx,
                pca_rank=pca_rank, chunk_size=5, min_iter=10, max_iter=50, convergence_threshold=1e-3, mask = mask, mode="null+(for-null)",
            )

            # save semantic direction in h-space
            torch.save(u_modify, u_modify_path)
            torch.save(vT_modify, vT_modify_path)

            if null_space_projection:
                u_null, s_null, vT_null = self.local_encoder_decoder_pullback_zt(
                zt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, op=op, block_idx=block_idx,
                pca_rank=pca_rank_null, chunk_size=5, min_iter=10, max_iter=50, convergence_threshold=1e-3, mask = ~mask, mode="null+(for-null)",
                )
                
                torch.save(u_null, u_null_path)
                torch.save(vT_null, vT_null_path)

        # normalize u, vT
        if not null_space_projection:
            vT = vT_modify / vT_modify.norm(dim=1, keepdim=True)
        else:
            vT_null = vT_null[:pca_rank_null, :]
            vT = (vT_null.T @ (vT_null @ vT_modify.T)).T
            vT = vT_modify - vT
            vT = vT / vT.norm(dim=1, keepdim=True)

        original_zt = zt.clone()
        for pc_idx in range(vis_num_pc):
            zts = {
                -1: None,
                1: None,
            }
            self.EXP_NAME = f'Edit_zt-edit_{self.edit_t}T-pc_{pc_idx}_select_mask{mask_index}_null_space_projection_{null_space_projection}_null_space_rank_{pca_rank_null}'        
            for direction in [1, -1]:
                vk = direction*vT[pc_idx, :].view(-1, *zT.shape[1:])
                # edit zt along vk direction with **x-space guidance**
                zt_list = [original_zt.clone()]
                for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                    zt_edit = self.x_space_guidance_direct(
                        zt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                        single_edit_step=self.x_space_guidance_edit_step,
                    )
                    zt_list.append(zt_edit)
                zt = torch.cat(zt_list, dim=0)
                if vis_num == 1:
                    zt = zt[[0,-1],:]
                else:
                    zt = zt[::(zt.size(0) // vis_num)]
                zts[direction] = zt
                # zt -> z0
            zt = torch.cat([(zts[-1].flip(dims=[0]))[:-1], zts[1]], dim=0)

            self.DDIMforwardsteps(
                zt, t_start_idx=self.edit_t_idx, t_end_idx=-1, 
                for_prompt_emb=self.for_prompt_emb, 
                edit_prompt_emb=self.edit_prompt_emb, 
                null_prompt_emb=self.null_prompt_emb,
                mode="null+(for-null)")


    @torch.no_grad()
    def run_edit_null_space_projection_zt_semantic(
            self, op, block_idx, vis_num, mask_index = 0, vis_num_pc=1, vis_vT=False, pca_rank=50, edit_prompt=None, null_space_projection = False, pca_rank_null=50, 
        ):
        print(f'current experiment : op : {op}, block_idx : {block_idx}, vis_num : {vis_num}, vis_num_pc : {vis_num_pc}, pca_rank : {pca_rank}, edit_prompt : {edit_prompt}, null_space_projection = {null_space_projection}, pca_rank_null={pca_rank_null}')
        '''
        1. z0 -> zT -> zt -> z0 ; we edit latent variable zt
        2. get local basis of h-space (u) and x-space (v) by using the power method
        3. edit sample with x-space guidance
        '''
        # set edit prompt
        if edit_prompt is not None:
            self.edit_prompt = edit_prompt
            self.edit_prompt_emb = self._get_prompt_emb(self.edit_prompt)

        # set edit_t
        self.scheduler.set_timesteps(self.for_steps)

        # get latent code (zT -> zt)
        if self.dataset_name == 'Random':
            zT = torch.randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)
        
        self.EXP_NAME = "original"
        if (not os.path.exists(os.path.join(self.result_folder, "original.png"))) or (not os.path.exists(os.path.join(self.result_folder, "mask/mask.pt"))):
            print("Generating images and creating masks......")
            _, x0 = self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=-1, 
                                          for_prompt_emb=self.for_prompt_emb, 
                                          edit_prompt_emb=self.edit_prompt_emb, 
                                          null_prompt_emb=self.null_prompt_emb,
                                          mode="null+(for-null)")
            masks = self.sam.mask_segmentation(Image.fromarray(np.array(x0[0].detach().cpu())), resolution=512)

        else:
            print("Loading masks......")
            masks = torch.load(os.path.join(self.result_folder, "mask/mask.pt"))
        
        if self.sampling_mode:
            return None
        mask = masks[mask_index].squeeze(dim=0).repeat(3, 1, 1)
        
        zt, t, t_idx = self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=self.edit_t_idx, 
                                            for_prompt_emb=self.for_prompt_emb, 
                                            edit_prompt_emb=self.edit_prompt_emb, 
                                            null_prompt_emb=self.null_prompt_emb,
                                            mode="null+(for-null)")
        assert t_idx == self.edit_t_idx



        # get local basis
        if not self.use_sega:        
            save_dir = os.path.join(self.result_folder, "basis", f'local_basis-{self.edit_t}T-"{self.edit_prompt}"-pca-rank-{pca_rank}-select-mask{mask_index}')
            os.makedirs(save_dir, exist_ok=True)
            u_modify_path = os.path.join(save_dir, f'u-modify.pt')
            vT_modify_path = os.path.join(save_dir, f'vT-modify.pt')
            u_null_path = os.path.join(save_dir, f'u-null-null_space_rank_{pca_rank_null}.pt')
            vT_null_path = os.path.join(save_dir, f'vT-null-null_space_rank_{pca_rank_null}.pt')        
            # load pre-computed local basis
            if os.path.exists(u_modify_path) and os.path.exists(vT_modify_path) and os.path.exists(u_null_path) and os.path.exists(vT_null_path):
                u_modify = torch.load(u_modify_path, map_location=self.device).type(self.dtype)
                vT_modify = torch.load(vT_modify_path, map_location=self.device).type(self.dtype)
                u_null = torch.load(u_null_path, map_location=self.device).type(self.dtype)
                vT_null = torch.load(vT_null_path, map_location=self.device).type(self.dtype)

            else:
                print('!!!RUN LOCAL PULLBACK!!!')
                zt = zt.to(device=self.device, dtype=self.dtype)

                vT_modify = self.get_delta_zt_via_grad(zt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, mask = mask, mode = self.tilda_v_score_type)

                torch.save(vT_modify, vT_modify_path)

                if null_space_projection:
                    u_null, s_null, vT_null = self.local_encoder_decoder_pullback_zt(
                    zt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, op=op, block_idx=block_idx,
                    pca_rank=pca_rank_null, chunk_size=5, min_iter=10, max_iter=50, convergence_threshold=1e-3, mask = ~mask, mode="null+(for-null)",
                    )
                    
                    torch.save(u_null, u_null_path)
                    torch.save(vT_null, vT_null_path)

            # normalize u, vT
            if not null_space_projection:
                vT = vT_modify / vT_modify.norm(dim=1, keepdim=True)
            else:
                vT_null = vT_null[:pca_rank_null, :]
                vT = (vT_null.T @ (vT_null @ vT_modify.T)).T
                vT = vT_modify - vT
                vT = vT / vT.norm(dim=1, keepdim=True)

            original_zt = zt.clone()
            for pc_idx in range(vis_num_pc):
                zts = {
                    -1: None,
                    1: None,
                }
                self.EXP_NAME = f'Edit_zt-edit_{self.edit_t}T-{op}-block_{block_idx}-pc_{pc_idx:0=3d}_pos-edit_prompt-{self.edit_prompt}_select_mask{mask_index}_null_space_projection_{null_space_projection}_null_space_rank_{pca_rank_null}_{self.tilda_v_score_type}'      
                for direction in [1, -1]:
                    vk = direction*vT[pc_idx, :].view(-1, *zT.shape[1:])
                    # edit zt along vk direction with **x-space guidance**
                    zt_list = [original_zt.clone()]
                    for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                        zt_edit = self.x_space_guidance_direct(
                            zt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                            single_edit_step=self.x_space_guidance_edit_step,
                        )
                        zt_list.append(zt_edit)
                    zt = torch.cat(zt_list, dim=0)
                    if vis_num == 1:
                        zt = zt[[0,-1],:]
                    else:
                        zt = zt[::(zt.size(0) // vis_num)]
                    zts[direction] = zt
                    # zt -> z0
                zt = torch.cat([(zts[-1].flip(dims=[0]))[:-1], zts[1]], dim=0)

            self.DDIMforwardsteps(
                zt, t_start_idx=self.edit_t_idx, t_end_idx=-1, 
                for_prompt_emb=self.for_prompt_emb, 
                edit_prompt_emb=self.edit_prompt_emb, 
                null_prompt_emb=self.null_prompt_emb,
                mode="null+(for-null)")
        else:
            self.EXP_NAME = f'sega-edit_prompt-{self.edit_prompt}'
            self.DDIMforwardsteps(
                zt, t_start_idx=self.edit_t_idx, t_end_idx=-1, 
                for_prompt_emb=self.for_prompt_emb, 
                edit_prompt_emb=self.edit_prompt_emb, 
                null_prompt_emb=self.null_prompt_emb,
                mode="null+(for-null)+(edit-null)")


    @torch.no_grad()
    def x_space_guidance_direct(self, zt, t_idx, vk, single_edit_step):
        # necesary parameters
        t = self.scheduler.timesteps[t_idx]

        # edit xt with vk
        zt_edit = zt + self.x_space_guidance_scale * single_edit_step * vk

        return zt_edit

    # utils
    def _get_prompt_emb(self, prompt):
        prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device = self.device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = False,
        )[0]
        return prompt_embeds



class EditDeepFloydIF(object):
    def __init__(self, args):
        # default setting
        self.seed = args.seed
        self.buffer_device  = args.buffer_device
        self.memory_bound   = args.memory_bound
        model_size = args.model_name.split("-")[2]
        # path
        self.result_folder = os.path.join(args.result_folder, f"for_prompt_{args.for_prompt}_cfg{args.guidance_scale}_seed{args.seed}_{model_size}")
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.obs_folder = args.obs_folder

        # get model
        self.stage_1, self.stage_2, self.stage_3 = get_DeepFloyd_IF_model(args)
        # self.vae  = self.pipe.vae
        self.unet = self.stage_1.unet
        
        self.mask_type = args.mask_type
        if self.mask_type == "SAM":
            self.sam = SAM(args, log_dir = self.result_folder)

        self.dtype  = args.dtype
        self.device = args.device

        # args (diffusion schedule)
        self.scheduler = get_deepfloyd_if_scheduler(args, self.stage_1.scheduler)
        self.for_steps = args.for_steps
        self.use_yh_custom_scheduler = args.use_yh_custom_scheduler

        # get image
        self.c_in = args.c_in
        self.image_size = args.image_size
        self.dataset = get_dataset(args)
        self.dataset_name = args.dataset_name

        # args (prompt)
        self.for_prompt         = args.for_prompt 

        self.neg_prompt         = args.neg_prompt 

        self.null_prompt        = ""
        self.inv_prompt         = args.inv_prompt if len(args.inv_prompt.split(',')[0]) <= 3 else ','.join([args.inv_prompt.split(',')[0]])

        self.for_prompt_emb     = self._get_prompt_emb(args.for_prompt)
        self.neg_prompt_emb     = self._get_prompt_emb(args.neg_prompt)
        self.null_prompt_emb    = self._get_prompt_emb("")
        self.inv_prompt_emb     = self._get_prompt_emb(args.inv_prompt)
        
        # args (guidance)
        self.guidance_scale     = args.guidance_scale
        self.guidance_scale_edit= args.guidance_scale_edit
        

        # args (h space edit)        
        self.edit_prompt_emb    = self._get_prompt_emb(args.edit_prompt)


        # x-space guidance
        self.x_edit_step_size = args.x_edit_step_size
        self.x_space_guidance_edit_step         = args.x_space_guidance_edit_step
        self.x_space_guidance_scale             = args.x_space_guidance_scale
        self.x_space_guidance_num_step          = args.x_space_guidance_num_step
        self.x_space_guidance_use_edit_prompt   = args.x_space_guidance_use_edit_prompt

        # args (h space edit + currently not using. please refer main.py)
        self.scheduler.set_timesteps(self.for_steps, device=self.device)
        self.edit_t         = args.edit_t
        self.edit_t_idx     = (self.scheduler.timesteps - self.edit_t * 1000).abs().argmin()
        self.no_edit_t         = args.no_edit_t
        self.no_edit_t_idx     = (self.scheduler.timesteps - self.no_edit_t * 1000).abs().argmin()
        self.sampling_mode  = args.sampling_mode
        self.tilda_v_score_type = args.tilda_v_score_type
        self.ablation_method = args.ablation_method
        self.vT_path = args.vT_path

    def _get_prompt_emb(self, prompt):

        prompt_embeds, _ = self.stage_1.encode_prompt(
            prompt,
            device = self.device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = False,
            negative_prompt = None,
        )

        return prompt_embeds

    def _classifer_free_guidance(self, latents, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode, do_classifier_free_guidance):
        assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)","(for-null)","(edit-null)","edit-proj[for](edit)","null+for+edit-proj[for](edit)"]


        if do_classifier_free_guidance:
            if mode == "null+(for-null)":
                model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)
            elif mode == "null+(for-null)+(edit-null)":
                model_input = torch.cat([latents] * 3, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), edit_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)
            elif mode == "null+(edit-null)":
                model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([edit_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)    
            elif mode == "(for-edit)":
                model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), edit_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)       
            elif mode == "(for-null)":
                model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)         
            elif mode == "(edit-null)":
                model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([edit_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)  
            elif mode == "edit-proj[for](edit)":       
                model_input = torch.cat([latents] * 2, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), edit_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)
            elif mode == "null+for+edit-proj[for](edit)":    
                model_input = torch.cat([latents] * 3, dim=0)
                prompt_emb = torch.cat([for_prompt_emb.repeat(latents.size(0), 1, 1), edit_prompt_emb.repeat(latents.size(0), 1, 1), null_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)          
        else:
            model_input = latents
            prompt_emb = for_prompt_emb.repeat(latents.size(0), 1, 1)

        noise_pred = self.unet(
            model_input, t,
            encoder_hidden_states=prompt_emb,
        ).sample

        # perform guidance
        if do_classifier_free_guidance:
            if mode == "null+(for-null)+(edit-null)":
                noise_pred_for, noise_pred_edit, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                noise_pred_for, predicted_variance = noise_pred_for.split(model_input.shape[1], dim=1)
                noise_pred_edit, _ = noise_pred_edit.split(model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_for - noise_pred_uncond) + self.guidance_scale_edit * (noise_pred_edit - noise_pred_uncond)
            elif mode == "null+(for-null)":
                noise_pred_for, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                noise_pred_for, predicted_variance = noise_pred_for.split(model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_for - noise_pred_uncond)
            elif mode == "null+(edit-null)":
                noise_pred_edit, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                noise_pred_edit, predicted_variance = noise_pred_edit.split(model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_edit - noise_pred_uncond)
            elif mode == "(for-edit)":
                noise_pred_for, noise_pred_edit = noise_pred.chunk(2)
                noise_pred_for, predicted_variance = noise_pred_for.split(model_input.shape[1], dim=1)
                noise_pred_edit, _ = noise_pred_edit.split(model_input.shape[1], dim=1)
                noise_pred = self.guidance_scale * (noise_pred_for - noise_pred_edit)   
            elif mode == "(for-null)":
                noise_pred_for, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred_for, predicted_variance = noise_pred_for.split(model_input.shape[1], dim=1)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                noise_pred = self.guidance_scale * (noise_pred_for - noise_pred_uncond)  
            elif mode == "(edit-null)":
                noise_pred_edit, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred_edit, predicted_variance = noise_pred_edit.split(model_input.shape[1], dim=1)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                noise_pred = self.guidance_scale * (noise_pred_edit - noise_pred_uncond)
            elif mode == "edit-proj[for](edit)":  
                noise_pred_for, noise_pred_edit = noise_pred.chunk(2)
                noise_pred_for, _ = noise_pred_for.split(model_input.shape[1], dim=1) 
                noise_pred_for = noise_pred_for - noise_pred_uncond
                noise_pred_edit, _ = noise_pred_edit.split(model_input.shape[1], dim=1)
                noise_pred_edit = noise_pred_edit - noise_pred_uncond
                noise_pred = get_prependicualr_component(noise_pred_edit, noise_pred_for) # projection in noise space
            elif mode == "null+for+edit-proj[for](edit)":
                noise_pred_for, noise_pred_edit, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                noise_pred_for, _ = noise_pred_for.split(model_input.shape[1], dim=1)
                noise_pred_for = noise_pred_for - noise_pred_uncond
                noise_pred_edit, _ = noise_pred_edit.split(model_input.shape[1], dim=1)
                noise_pred_edit = noise_pred_edit - noise_pred_uncond
                noise_pred_edit = get_prependicualr_component(noise_pred_edit, noise_pred_for) # projection in noise space
                noise_pred = noise_pred_uncond + self.guidance_scale * noise_pred_for + self.guidance_scale_edit * noise_pred_edit         
        return noise_pred

    @torch.no_grad()
    def superresolution(self, x0, cond_prompt, cond_prompt_emb, uncond_prompt_emb):
        bs = len(x0)
        stage_2_output = self.stage_2(
            image=x0,
            prompt_embeds=cond_prompt_emb.repeat(bs, 1, 1),
            negative_prompt_embeds=uncond_prompt_emb.repeat(bs, 1, 1),
            output_type="pt",
        ).images
        stage_2_pil = pt_to_pil(stage_2_output)
        concatenate_pil_horizontally(stage_2_pil).save(os.path.join(self.result_folder, f'{self.EXP_NAME}_stage2.png'))

        # stage 3
        # stage_3_output = self.stage_3(prompt=[cond_prompt] * bs, image=stage_2_output, noise_level=100).images
        # concatenate_pil_horizontally(stage_3_output).save(os.path.join(self.result_folder, f'{self.EXP_NAME}_stage3.png'))
        # return stage_2_pil[0], stage_3_output[0]

        return stage_2_pil[0], None

    @torch.no_grad()
    def mask_diffedit(self, x0, for_prompt_emb, edit_prompt_emb, null_prompt_emb):
        t = torch.tensor(500).cuda()
        at = extract(self.scheduler.alphas_cumprod, t, x0.shape)
        xt = at.sqrt() * x0 + (1 - at).sqrt() * torch.randn(10, self.c_in, self.image_size, self.image_size, dtype=self.dtype, device=self.device)
        eps_1 = self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode="null+(for-null)", do_classifier_free_guidance=True)
        eps_2 = self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode="null+(edit-null)", do_classifier_free_guidance=True)     
        mask = (eps_1 - eps_2).mean(dim=0, keepdim = True).mean(dim=1)
        mask = torch.round((mask - mask.min() / (mask.max() - mask.min()))).to(torch.bool)
        log_dir = os.path.join(self.result_folder, "mask")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tvu.save_image(mask, os.path.join(log_dir, f'mask_diffedit_t_{t.item()}.png'))
        return mask

    @torch.no_grad()
    def DDPMforwardsteps(
            self, xt, t_start_idx, t_end_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = "null+(for-null)", **kwargs
        ):
        '''
        Prompt
            (CFG)       pos : for_prompt, neg : neg_prompt
            (no CFG)    pos : for_prompt
        '''
        assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)"]
        # before start
        num_inference_steps = self.for_steps
        do_classifier_free_guidance = self.guidance_scale > 1.0
        memory_bound = self.memory_bound // 2 if do_classifier_free_guidance else self.memory_bound

        # set timestep (we do not use default scheduler set timestep method)
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        else:
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            
        #############################################
        # denoising loop (t_start_idx -> t_end_idx) #
        #############################################
        
        for t_idx, t in enumerate(self.scheduler.timesteps):
            # skip
            if (t_idx < t_start_idx): 
                continue
                
            # start sampling
            elif t_start_idx == t_idx:
                # print('t_start_idx : ', t_idx)
                pass

            # end sampling
            elif t_idx == t_end_idx:
                # print('t_end_idx : ', t_idx)
                return xt, t, t_idx

            # split xt to avoid OOM
            xt = xt.to(device=self.buffer_device)
            if xt.size(0) == 1:
                xt_buffer = [xt]
            else:
                xt_buffer = list(xt.chunk(xt.size(0) // memory_bound))

            # loop over buffer
            for buffer_idx, xt in enumerate(xt_buffer):
                # overload to device
                xt = xt.to(device=self.device)
                xt = self.scheduler.scale_model_input(xt, t)
                
                noise_pred = self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = mode, do_classifier_free_guidance = do_classifier_free_guidance)
                # compute the previous noisy sample x_t -> x_t-1
                xt = self.scheduler.step(
                    noise_pred, t, xt, eta=0
                ).prev_sample
                # save xt in buffer
                xt_buffer[buffer_idx] = xt.to(self.buffer_device)

            xt = torch.cat(xt_buffer, dim=0)
            xt = xt.to(device=self.device)
            
            del xt_buffer
            torch.cuda.empty_cache()

        # latents = 1 / 0.18215 * latents
        # x0 = self.vae.decode(latents).sample
        xt = (xt / 2 + 0.5).clamp(0, 1)
        tvu.save_image(xt, os.path.join(self.result_folder, f'{self.EXP_NAME}_stage1.png'), nrow = xt.size(0))
        xt = (xt * 255).to(torch.uint8).permute(0, 2, 3, 1)
        return xt


    @torch.no_grad()
    def MaskedDDPMforwardsteps(
            self, xt, t_start_idx, t_end_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask, **kwargs
        ):
        '''
        Prompt
            (CFG)       pos : for_prompt, neg : neg_prompt
            (no CFG)    pos : for_prompt
        '''
        mask = mask.to(dtype = torch.float16, device = xt.device)
        # before start
        num_inference_steps = self.for_steps
        do_classifier_free_guidance = self.guidance_scale > 1.0
        # cross_attention_kwargs      = None
        memory_bound = self.memory_bound // 2 if do_classifier_free_guidance else self.memory_bound

        # set timestep (we do not use default scheduler set timestep method)
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        else:
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            
        #############################################
        # denoising loop (t_start_idx -> t_end_idx) #
        #############################################
        
        for t_idx, t in enumerate(self.scheduler.timesteps):
            # skip
            if (t_idx < t_start_idx): 
                continue
                
            # start sampling
            elif t_start_idx == t_idx:
                # print('t_start_idx : ', t_idx)
                pass

            # end sampling
            elif t_idx == t_end_idx:
                # print('t_end_idx : ', t_idx)
                return xt, t, t_idx

            # split xt to avoid OOM
            xt = xt.to(device=self.buffer_device)
            if xt.size(0) == 1:
                xt_buffer = [xt]
            else:
                xt_buffer = list(xt.chunk(xt.size(0) // memory_bound))

            # loop over buffer
            for buffer_idx, xt in enumerate(xt_buffer):
                # overload to device
                xt = xt.to(device=self.device)
                xt = self.scheduler.scale_model_input(xt, t)
                
                noise_pred_for = self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = "null+(for-null)", do_classifier_free_guidance = do_classifier_free_guidance)
                xt_for = self.scheduler.step(
                    noise_pred_for, t, xt, eta=0
                ).prev_sample
                noise_pred_edit = self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = "null+(edit-null)", do_classifier_free_guidance = do_classifier_free_guidance)
                xt_edit = self.scheduler.step(
                    noise_pred_edit, t, xt, eta=0
                ).prev_sample              
                xt =  xt_edit * mask + xt_for * (1 - mask)
                # save xt in buffer
                xt_buffer[buffer_idx] = xt.to(self.buffer_device)

            xt = torch.cat(xt_buffer, dim=0)
            xt = xt.to(device=self.device)
            
            del xt_buffer
            torch.cuda.empty_cache()

        # latents = 1 / 0.18215 * latents
        # x0 = self.vae.decode(latents).sample
        xt = (xt / 2 + 0.5).clamp(0, 1)
        tvu.save_image(xt, os.path.join(self.result_folder, f'{self.EXP_NAME}_stage1.png'), nrow = xt.size(0))
        xt = (xt * 255).to(torch.uint8).permute(0, 2, 3, 1)
        return xt


    def get_x0(self, xt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = None, mode = "null+(for-null)+(edit-null)", flatten = False):
        # assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)"]
        
        do_classifier_free_guidance = self.guidance_scale > 1.0

        noise_pred = self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = mode, do_classifier_free_guidance = do_classifier_free_guidance)
        
        at = extract(self.scheduler.alphas_cumprod, t, xt.shape)

        x0_hat = (xt - noise_pred * (1 - at).sqrt()) / at.sqrt()

        if mask is not None:
            x0_hat = x0_hat[:, mask]
            # print("x0_hat", x0_hat.shape)
            return x0_hat

        # only use this for single xt and no mask:
        if flatten:
            # print(xt.shape, x0_hat.shape)
            c_i, w_i, h_i = xt.size(1), xt.size(2), xt.size(3)
            x0_hat = x0_hat.view(-1, c_i*w_i*h_i)
        return x0_hat

    def local_encoder_decoder_pullback_xt(
            self, xt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, op=None, block_idx=None,
            pca_rank=50, chunk_size=25, min_iter=10, max_iter=100, convergence_threshold=1e-3, mask = None,
            mode = "null+(for-null)+(edit-null)"
        ):
        '''
        Args
            - xt : xt
            - op : ['down', 'mid', 'up']
            - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
            - pooling : ['pixel-sum', 'channel-sum', 'single-channel', 'multiple-channel']
        Returns
            - h : hidden feature
        '''
        assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)"]
        num_chunk = pca_rank // chunk_size if pca_rank % chunk_size == 0 else pca_rank // chunk_size + 1
        # print(pca_rank, chunk_size, num_chunk) # 2, 25, 1

        # get h samples
        time_s = time.time()

        c_i, w_i, h_i = xt.size(1), xt.size(2), xt.size(3)
        # c_o, w_o, h_o = h_shape[1], h_shape[2], h_shape[3]
        if mask is None:
            c_o, w_o, h_o = c_i, w_i, h_i # output shape of x^0
        else:
            l_o = mask.sum().item()


        a = torch.tensor(0., device=xt.device, dtype=xt.dtype)

        # Algorithm 1
        vT = torch.randn(c_i*w_i*h_i, pca_rank, device=xt.device, dtype=torch.float)
        vT, _ = torch.linalg.qr(vT)
        v = vT.T
        v = v.view(-1, c_i, w_i, h_i)


        time_s = time.time()

        # Jacobian subspace iteration
        for i in range(max_iter):
            v = v.to(device=xt.device, dtype=xt.dtype)
            v_prev = v.detach().cpu().clone()
            
            u = []
            v_buffer = list(v.chunk(num_chunk))
            for vi in v_buffer:
                g = lambda alpha : self.get_x0(xt + alpha*vi, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = mask, mode=mode)
                ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a) # ui = J@vi
                u.append(ui.detach().cpu().clone())
            u = torch.cat(u, dim=0)
            u = u.to(xt.device, xt.dtype)


            if mask is None:
                g = lambda xt : einsum(
                    u, self.get_x0(xt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask=mask, mode=mode), 'b c w h, i c w h -> b'
                )
            else:
                g = lambda xt : einsum(
                    u, self.get_x0(xt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask=mask, mode=mode), 'b l, i l -> b'
                )                
            v_ = torch.autograd.functional.jacobian(g, xt) # vi = ui.T@J
            v_ = v_.view(-1, c_i*w_i*h_i).to(torch.float)

            _, s, v = torch.linalg.svd(v_, full_matrices=False)
            v = v.view(-1, c_i, w_i, h_i).to(dtype=xt.dtype)
            if mask is None:
                u = u.view(-1, c_o, w_o, h_o)
            else:
                u = u.view(-1, l_o)
            
            convergence = torch.dist(v_prev, v.detach().cpu()).item()
            print(f'power method : {i}-th step convergence : ', convergence)
            
            if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
                print('reach convergence threshold : ', convergence)
                break

        time_e = time.time()
        print('power method runtime ==', time_e - time_s)

        if mask is None:
            u, s, vT = u.reshape(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()
        else:
            u, s, vT = u.reshape(-1, l_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()
        return u, s, vT


    @torch.no_grad()
    def get_delta_xt_via_grad(self, xt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = None, mode = "null+(for-null)+(edit-null)"):
        # assert mode in ["null+(for-null)+(edit-null)","null+(for-null)","null+(edit-null)","(for-edit)"]
        
        do_classifier_free_guidance = self.guidance_scale > 1.0

        noise_pred = self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = "null+(for-null)", do_classifier_free_guidance = do_classifier_free_guidance)
        at = extract(self.scheduler.alphas_cumprod, t, xt.shape)
        x0_hat = (xt - noise_pred * (1 - at).sqrt()) / at.sqrt()

        noise_pred_after= self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = mode, do_classifier_free_guidance = do_classifier_free_guidance)
        at = extract(self.scheduler.alphas_cumprod, t, xt.shape)
        x0_hat_after = (xt - noise_pred_after * (1 - at).sqrt()) / at.sqrt()

        x0_hat_delta = x0_hat_after - x0_hat
        c_i, w_i, h_i = xt.size(1), xt.size(2), xt.size(3)

        delta_noise_norm = (noise_pred_after - noise_pred).norm()
        delta_x0_norm = x0_hat_delta.norm()

        # we can even add mask here
        if mask is not None:
            x0_hat_delta_flat = x0_hat_delta[:,mask]
            # print(x0_hat_delta_flat.shape)
        else:
            x0_hat_delta_flat = x0_hat_delta.view(-1,c_i*w_i*h_i)

        g = lambda v : torch.sum(x0_hat_delta_flat * self.get_x0(v, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = mask, mode = mode, flatten=True))

        v_ = torch.autograd.functional.jacobian(g, xt)


        v_ = v_.view(-1, c_i*w_i*h_i)
        v_ = v_ / v_.norm(dim=1, keepdim=True)

        xt_delta = v_.view(-1, c_i, w_i, h_i)
        xt_new = 10.0 * xt_delta + xt
        noise_pred_new = self._classifer_free_guidance(xt_new, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode = "null+(for-null)", do_classifier_free_guidance = do_classifier_free_guidance)
        x0_hat_new = (xt_new - noise_pred_new * (1 - at).sqrt()) / at.sqrt()
        # tvu.save_image((x0_hat_new / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'viagrad_x0_hat.png'))
        
        return v_

    @torch.no_grad()
    def get_v_modify(self, xt, t, t_idx, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mask = None, mode= ["(for-edit)-direct"], jacobian=False):
        # find direction using jacobian, have different mode
        if jacobian:
            vT_modify = self.get_delta_xt_via_grad(xt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, mask = mask, mode = self.tilda_v_score_type)
            return vT_modify

        # find direction directly, only having 3 input mode
        if mode == "(for-edit)-direct":
            vT_modify = self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode="(for-edit)", do_classifier_free_guidance=True)
            vT_modify = vT_modify.reshape(1, -1)
        elif mode == "(edit-null)-direct":
            vT_modify = self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode="(edit-null)", do_classifier_free_guidance=True)
            vT_modify = -vT_modify.reshape(1, -1)
        elif mode == "proj_null[for-null](edit-null)-direct":
            eps_1 = self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode="(for-null)", do_classifier_free_guidance=True).reshape(1, -1)
            eps_2 = self._classifer_free_guidance(xt, t, for_prompt_emb, edit_prompt_emb, null_prompt_emb, mode="(edit-null)", do_classifier_free_guidance=True).reshape(1, -1)
            vT_modify = eps_2 - ((eps_2 * eps_1).sum()/(eps_1 * eps_1).sum()) * eps_1
            vT_modify = -vT_modify.reshape(1, -1)
        return vT_modify


    @torch.no_grad()
    def run_edit_null_space_projection_xt(
            self, op, block_idx, vis_num, mask_index = 0, vis_num_pc=1, vis_vT=False, pca_rank=50, edit_prompt=None, null_space_projection = False, pca_rank_null=50, 
        ):
        print(f'current experiment : op : {op}, block_idx : {block_idx}, vis_num : {vis_num}, vis_num_pc : {vis_num_pc}, pca_rank : {pca_rank}, edit_prompt : {edit_prompt}, null_space_projection = {null_space_projection}, pca_rank_null={pca_rank_null}')
        '''
        1. x0 -> xT -> xt -> x0 ; we edit latent variable xt
        2. get local basis of h-space (u) and x-space (v) by using the power method
        3. edit sample with x-space guidance
        '''
        # set edit prompt
        if edit_prompt is not None:
            self.edit_prompt = edit_prompt
            self.edit_prompt_emb = self._get_prompt_emb(self.edit_prompt)

        # set edit_t
        self.scheduler.set_timesteps(self.for_steps)

        # get latent code (xT -> xt)
        if self.dataset_name == 'Random':
            xT = torch.randn(1, self.c_in, self.image_size, self.image_size, dtype=self.dtype, device=self.device)
        self.EXP_NAME = "original"
        if (not os.path.exists(os.path.join(self.result_folder, "original_stage1.png"))) or (not os.path.exists(os.path.join(self.result_folder, "mask/mask.pt"))):
            print("Generating images and creating masks......")
            x0 = self.DDPMforwardsteps(xT, t_start_idx=0, t_end_idx=-1, 
                                        for_prompt_emb=self.for_prompt_emb, 
                                        edit_prompt_emb=self.edit_prompt_emb, 
                                        null_prompt_emb=self.null_prompt_emb,
                                        mode="null+(for-null)")
            x0_stage2_pil, x0_stage3_pil = self.superresolution([Image.fromarray(x0[0].detach().cpu().numpy())], self.for_prompt, self.for_prompt_emb, self.null_prompt_emb)
            masks = self.sam.mask_segmentation(x0_stage2_pil, resolution = self.image_size)
        else:
            print("Loading masks......")
            masks = torch.load(os.path.join(self.result_folder, "mask/mask.pt"))
        if self.sampling_mode:
            return None
        mask = masks[mask_index].squeeze(dim=0).repeat(3, 1, 1)
        
        xt, t, t_idx = self.DDPMforwardsteps(xT, t_start_idx=0, t_end_idx=self.edit_t_idx, 
                                            for_prompt_emb=self.for_prompt_emb, 
                                            edit_prompt_emb=self.edit_prompt_emb, 
                                            null_prompt_emb=self.null_prompt_emb,
                                            mode="null+(for-null)")
        assert t_idx == self.edit_t_idx

        save_dir = os.path.join(self.result_folder, "basis", f'local_basis-{self.edit_t}T-pca-rank-{pca_rank}-select-mask{mask_index}')
        os.makedirs(save_dir, exist_ok=True)
        u_modify_path = os.path.join(save_dir, f'u-modify.pt')
        vT_modify_path = os.path.join(save_dir, f'vT-modify.pt')
        u_null_path = os.path.join(save_dir, f'u-null-null_space_rank_{pca_rank_null}.pt')
        vT_null_path = os.path.join(save_dir, f'vT-null-null_space_rank_{pca_rank_null}.pt')        
        # load pre-computed local basis
        if os.path.exists(u_modify_path) and os.path.exists(vT_modify_path) and os.path.exists(u_null_path) and os.path.exists(vT_null_path):
            print('!!!Load CALCULATED BASIS!!!')
            u_modify = torch.load(u_modify_path, map_location=self.device).type(self.dtype)
            vT_modify = torch.load(vT_modify_path, map_location=self.device).type(self.dtype)
            u_null = torch.load(u_null_path, map_location=self.device).type(self.dtype)
            vT_null = torch.load(vT_null_path, map_location=self.device).type(self.dtype)

        else:
            print('!!!RUN LOCAL PULLBACK!!!')
            xt = xt.to(device=self.device, dtype=self.dtype)

            u_modify, s_modify, vT_modify = self.local_encoder_decoder_pullback_xt(
                xt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, op=op, block_idx=block_idx,
                pca_rank=pca_rank, chunk_size=5, min_iter=10, max_iter=50, convergence_threshold=1e-3, mask = mask, mode="null+(for-null)",
            )

            # save semantic direction in h-space
            torch.save(u_modify, u_modify_path)
            torch.save(vT_modify, vT_modify_path)

            if null_space_projection:
                u_null, s_null, vT_null = self.local_encoder_decoder_pullback_xt(
                xt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, op=op, block_idx=block_idx,
                pca_rank=pca_rank_null, chunk_size=5, min_iter=10, max_iter=50, convergence_threshold=1e-3, mask = ~mask, mode="null+(for-null)",
                )
                
                torch.save(u_null, u_null_path)
                torch.save(vT_null, vT_null_path)
        

        # normalize u, vT
        if not null_space_projection:
            vT = vT_modify / vT_modify.norm(dim=1, keepdim=True)
        else:
            vT_null = vT_null[:pca_rank_null, :]
            vT = (vT_null.T @ (vT_null @ vT_modify.T)).T
            vT = vT_modify - vT
            vT = vT / vT.norm(dim=1, keepdim=True)


        original_xt = xt.clone()
        for pc_idx in range(vis_num_pc):
            xts = {
                -1: None,
                1: None,
            }
            self.EXP_NAME = f'Non-semantic_Edit_xt-edit_{self.edit_t}T-select_mask{mask_index}-edit_space_rank-{pc_idx}-null_space_projection_{null_space_projection}-null_space_rank_{pca_rank_null}_{self.tilda_v_score_type}'
                       
            for direction in [1, -1]:
                vk = direction*vT[pc_idx, :].view(-1, *xT.shape[1:])
                # edit xt along vk direction with **x-space guidance**
                xt_list = [original_xt.clone().to(torch.device('cuda:0'))]
                for _ in range(self.x_space_guidance_num_step):
                    xt_edit = self.x_space_guidance_direct(
                        xt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                        single_edit_step=self.x_space_guidance_edit_step,
                    )
                    xt_list.append(xt_edit)
                xt = torch.cat(xt_list, dim=0)
                # xt = xt[::(xt.size(0) // vis_num)]
                if vis_num == 1:
                    xt = xt[[0,-1],:]
                else:
                    xt = xt[::(xt.size(0) // vis_num)]
                xts[direction] = xt
                # xt -> z0
            xt = torch.cat([(xts[-1].flip(dims=[0]))[:-1], xts[1]], dim=0)

            x0 = self.DDPMforwardsteps(
                xt, t_start_idx=self.edit_t_idx, t_end_idx=-1, 
                for_prompt_emb=self.for_prompt_emb, 
                edit_prompt_emb=self.edit_prompt_emb, 
                null_prompt_emb=self.null_prompt_emb,
                mode="null+(for-null)")
            self.superresolution([Image.fromarray(x.detach().cpu().numpy()) for x in x0], self.for_prompt, self.for_prompt_emb, self.null_prompt_emb)


    @torch.no_grad()
    def run_edit_null_space_projection_xt_semantic(
            self, op, block_idx, vis_num, mask_index = 0, vis_num_pc=1, vis_vT=False, pca_rank=50, edit_prompt=None, null_space_projection = False, pca_rank_null=50, jacobian = False
        ):
        print(f'current experiment : op : {op}, block_idx : {block_idx}, vis_num : {vis_num}, vis_num_pc : {vis_num_pc}, pca_rank : {pca_rank}, edit_prompt : {edit_prompt}, null_space_projection = {null_space_projection}, pca_rank_null={pca_rank_null}')
        '''
        1. x0 -> xT -> xt -> x0 ; we edit latent variable xt
        2. get local basis of h-space (u) and x-space (v) by using the power method
        3. edit sample with x-space guidance
        '''
        # set edit prompt
        if edit_prompt is not None:
            self.edit_prompt = edit_prompt
            self.edit_prompt_emb = self._get_prompt_emb(self.edit_prompt)

        # set edit_t
        self.scheduler.set_timesteps(self.for_steps)

        # get latent code (xT -> xt)
        if self.dataset_name == 'Random':
            xT = torch.randn(1, self.c_in, self.image_size, self.image_size, dtype=self.dtype, device=self.device)


        if self.mask_type == "SAM":
            if (not os.path.exists(os.path.join(self.result_folder, "original_stage1.png"))) or (not os.path.exists(os.path.join(self.result_folder, "mask/mask.pt"))):
                print("Generating images and creating masks......")
                self.EXP_NAME = "original"
                x0 = self.DDPMforwardsteps(xT, t_start_idx=0, t_end_idx=-1, 
                                            for_prompt_emb=self.for_prompt_emb, 
                                            edit_prompt_emb=self.edit_prompt_emb, 
                                            null_prompt_emb=self.null_prompt_emb,
                                            mode="null+(for-null)")
                x0_stage2_pil, x0_stage3_pil = self.superresolution([Image.fromarray(x0[0].detach().cpu().numpy())], self.for_prompt, self.for_prompt_emb, self.null_prompt_emb)
                masks = self.sam.mask_segmentation(x0_stage2_pil, resolution = self.image_size)
            else:
                print("Loading masks......")
                masks = torch.load(os.path.join(self.result_folder, "mask/mask.pt"))
            mask = masks[mask_index].squeeze(dim=0).repeat(3, 1, 1)
        elif self.mask_type == "diffedit":
            self.EXP_NAME = "original"
            x0 = self.DDPMforwardsteps(xT, t_start_idx=0, t_end_idx=-1, 
                                        for_prompt_emb=self.for_prompt_emb, 
                                        edit_prompt_emb=self.edit_prompt_emb, 
                                        null_prompt_emb=self.null_prompt_emb,
                                        mode="null+(for-null)")
            x0_stage2_pil, x0_stage3_pil = self.superresolution([Image.fromarray(x0[0].detach().cpu().numpy())], self.for_prompt, self.for_prompt_emb, self.null_prompt_emb)
            mask = self.mask_diffedit((x0.to(torch.float16) * 2/ 255 - 1).permute(0, 3, 1, 2), self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb)
        
        if self.sampling_mode:
            return None
        xt, t, t_idx = self.DDPMforwardsteps(xT, t_start_idx=0, t_end_idx=self.edit_t_idx, 
                                            for_prompt_emb=self.for_prompt_emb, 
                                            edit_prompt_emb=self.edit_prompt_emb, 
                                            null_prompt_emb=self.null_prompt_emb,
                                            mode="null+(for-null)")
        assert t_idx == self.edit_t_idx

        save_dir = os.path.join(self.result_folder, "basis")
        os.makedirs(save_dir, exist_ok=True)
        # get local basis
        if self.ablation_method == "null-space-proj":

            if not os.path.exists(self.vT_path):

                vT_modify = self.get_v_modify(xt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, mask = mask, mode= self.tilda_v_score_type, jacobian = jacobian)
                # save semantic direction in h-space

                if null_space_projection:
                    print('!!!RUN LOCAL PULLBACK!!!')
                    _, _, vT_null = self.local_encoder_decoder_pullback_xt(
                    xt, t, t_idx, self.for_prompt_emb, self.edit_prompt_emb, self.null_prompt_emb, op=op, block_idx=block_idx,
                    pca_rank=pca_rank_null, chunk_size=5, min_iter=10, max_iter=50, convergence_threshold=1e-3, mask = ~mask, mode="null+(for-null)",
                    )
                    vT_null = vT_null[:pca_rank_null, :]
                    vT = (vT_null.T @ (vT_null @ vT_modify.T)).T
                    vT = vT_modify - vT
                else:
                    vT = vT_modify

                vT = vT / vT.norm(dim=1, keepdim=True)

                BASIS_NAME = f"edit-{self.edit_t}T-edit_prompt-{self.edit_prompt}-select_mask{mask_index}-null_space_projection_{null_space_projection}_null_space_rank_{pca_rank_null}_{self.tilda_v_score_type}"
                print(BASIS_NAME)
                for pc_idx in range(min(vT.shape[0], vis_num_pc)):
                    self.EXP_NAME = f'Semantic_Edit_xt-{BASIS_NAME}-pc_{pc_idx:0=3d}'
                    vT_path = os.path.join(save_dir, f'{self.EXP_NAME}-vT.pt')
                    print(vT_path)
                    torch.save(vT[[pc_idx], :], vT_path)        
            else:
                print('!!!LOAD VT FROM VT_PATH!!!')
                vT = torch.load(self.vT_path)
                BASIS_NAME = f"load-basis-'{os.path.basename(self.vT_path)}'"


            original_xt = xt.clone()
            for pc_idx in range(vis_num_pc):
                xts = {
                    -1: None,
                    1: None,
                }
                self.EXP_NAME = f'Semantic_Edit_xt-{BASIS_NAME}_scale_{self.x_space_guidance_scale}'           
                for direction in [1, -1]:
                    vk = direction*vT[pc_idx, :].view(-1, *xT.shape[1:])
                    xt_list = [original_xt.clone().to(torch.device('cuda:0'))]
                    for _ in range(self.x_space_guidance_num_step):
                        xt_edit = self.x_space_guidance_direct(
                            xt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                            single_edit_step=self.x_space_guidance_edit_step,
                        )
                        xt_list.append(xt_edit)
                    xt = torch.cat(xt_list, dim=0)
                    if vis_num == 1:
                        xt = xt[[0,-1],:]
                    else:
                        xt = xt[::(xt.size(0) // vis_num)]
                    xts[direction] = xt
                    # xt -> z0
                xt = torch.cat([(xts[-1].flip(dims=[0]))[:-1], xts[1]], dim=0)

            x0 = self.DDPMforwardsteps(
                xt, t_start_idx=self.edit_t_idx, t_end_idx=-1, 
                for_prompt_emb=self.for_prompt_emb, 
                edit_prompt_emb=self.edit_prompt_emb, 
                null_prompt_emb=self.null_prompt_emb,
                mode="null+(for-null)")

            
        elif self.ablation_method == "sega":
            self.EXP_NAME = f'sega-edit_prompt-{self.edit_prompt}-mask_type-{self.mask_type}-select_mask{mask_index}'
            x0 = self.DDPMforwardsteps(
                xt, t_start_idx=self.edit_t_idx, t_end_idx=-1, 
                for_prompt_emb=self.for_prompt_emb, 
                edit_prompt_emb=self.edit_prompt_emb, 
                null_prompt_emb=self.null_prompt_emb,
                mode="null+(for-null)+(edit-null)")

        elif self.ablation_method == "diffedit":
            self.EXP_NAME = f'diffedit-edit_prompt-{self.edit_prompt}-mask_type-{self.mask_type}-select_mask{mask_index}'
            x0 = self.MaskedDDPMforwardsteps(
                xt, t_start_idx=self.edit_t_idx, t_end_idx=-1, 
                for_prompt_emb=self.for_prompt_emb, 
                edit_prompt_emb=self.edit_prompt_emb, 
                null_prompt_emb=self.null_prompt_emb, 
                mask=mask)
            
        self.superresolution([Image.fromarray(x.detach().cpu().numpy()) for x in x0], self.for_prompt, self.for_prompt_emb, self.null_prompt_emb)

    @torch.no_grad()
    def x_space_guidance_direct(self, xt, t_idx, vk, single_edit_step):
        # necesary parameters
        t = self.scheduler.timesteps[t_idx]

        # edit xt with vk
        xt_edit = xt + self.x_space_guidance_scale * single_edit_step * vk

        return xt_edit


################
# Uncond model #
################
class EditUncondDiffusion(object):
    def __init__(self, args):
        # default setting
        self.pca_device     = args.pca_device
        self.buffer_device  = args.buffer_device
        self.memory_bound   = args.memory_bound
        self.device = args.device
        self.dtype = args.dtype
        self.seed = args.seed
        self.save_result_as = args.save_result_as

        # get model
        self.unet = get_custom_diffusion_model(args)
        self.scheduler = get_custom_diffusion_scheduler(args)
        self.model_name = args.model_name

        if 'HF' in self.model_name:
            self.scheduler = self.unet.scheduler if self.scheduler is None else self.scheduler
            self.unet = self.unet.unet

        # args (model)        
        self.image_size = args.image_size
        self.c_in = 3

        # get image
        self.dataset = get_dataset(args)
        self.dataset_name = args.dataset_name

        # args (diffusion schedule)
        self.for_steps = args.for_steps
        self.inv_steps = args.inv_steps
        self.use_yh_custom_scheduler = args.use_yh_custom_scheduler

        # args (edit)
        
        self.edit_t     = args.edit_t

        self.scheduler.set_timesteps(self.for_steps, device=self.device)
        self.edit_t_idx     = (self.scheduler.timesteps - self.edit_t * 1000).abs().argmin()
        self.performance_boosting_t_idx = (self.scheduler.timesteps - args.performance_boosting_t * 1000).abs().argmin() if args.performance_boosting_t > 0 else 1000
        print(f'performance_boosting_t_idx: {self.performance_boosting_t_idx}')
        

        # args (x-space guidance)
        self.use_x_space_guidance           = args.use_x_space_guidance
        self.x_space_guidance_edit_step     = args.x_space_guidance_edit_step
        self.x_space_guidance_scale         = args.x_space_guidance_scale
        self.x_space_guidance_num_step      = args.x_space_guidance_num_step
        
        # path
        if self.dataset_name in ["CelebA_HQ_mask", "FFHQ", "AFHQ", "Metface", "Flower", "LSUN_church", "LSUN_bedroom"]:
            self.result_folder = os.path.join(args.result_folder, f"sample_idx{args.sample_idx}")
        elif self.dataset_name == "Random":
            self.result_folder = os.path.join(args.result_folder, f"sample_seed{args.seed}")

        self.mask_type = args.mask_type
        if self.mask_type == "SAM":
            self.sam = SAM(args, log_dir = self.result_folder)

        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.obs_folder    = args.obs_folder
        self.vT_path = args.vT_path

        self.vT1_path = args.vT1_path
        # self.vT2_path = args.vT2_path

        self.args = args

    @torch.no_grad()
    def run_DDIMforward(self, num_samples=5):
        print('start DDIMforward')
        self.EXP_NAME = f'DDIMforward'

        # get latent code
        xT = torch.randn(num_samples, self.c_in, self.image_size, self.image_size, device=self.device, dtype=self.dtype)
        print('shape of xT : ', xT.shape)
        print('norm of xT : ', xT.view(num_samples, -1).norm(dim=-1))

        # simple DDIMforward
        self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=-1, vis_psd=False)

    @torch.no_grad()
    def run_DDIMinversion(self, idx):
        '''
        Args
            - img idx
        Returns
            - zT
            - zt_traj
            - et_traj
        '''
        print('start DDIMinversion')
        EXP_NAME = f'DDIMinversion-{self.dataset_name}_{idx}'

        # before start
        num_inference_steps = self.inv_steps        
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, is_inversion=True)
        else:
            raise ValueError('please set use_yh_custom_scheduler = True')
        timesteps = self.scheduler.timesteps

        # get image
        x0 = self.dataset[idx]
        tvu.save_image((x0 / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'original.png'))

        ##################
        # denoising loop #
        ##################
        xt = x0.to(self.device, dtype=self.dtype)

        for i, t in enumerate(timesteps):
            if i == len(timesteps) - 1:
                break

            # 1. predict noise model_output
            et = self.unet(xt, t)
            if not isinstance(et, torch.Tensor):
                et = et.sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            xt = self.scheduler.step(
                et, t, xt, eta=0, use_clipped_model_output=None, 
            ).prev_sample

        # visualize latents, zt_traj, et_traj
        tvu.save_image(
            (xt / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'xT-{EXP_NAME}.png'),
        )

        return xt

    
    @torch.no_grad()
    def group_edit_null_space_projection(
            self, idx, 
            **kwargs,
        ):
        
        # get latent code
        if self.dataset_name == 'Random':
            xT = torch.randn(1, 3, 256, 256, dtype=self.dtype, device=self.device)
        else:
            xT = self.run_DDIMinversion(idx=idx)

        mask = self.dataset.getmask(idx = self.args.sample_idx, choose_sem = self.args.choose_sem)

        # xT -> xt
        xt, t, t_idx = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=self.edit_t_idx)
        assert t_idx == self.edit_t_idx

        vT_list = []
        print('!!!LOAD VT FROM VT_PATH!!!')
        vT0 = torch.load(self.vT_path)
        vT1 = torch.load(self.vT1_path)

        BASIS_NAME = f"load-basis-2"
        vT_list.append(vT0)
        vT_list.append(vT1)

        # get latent code
        original_xt = xt.detach()
        xt_temp = original_xt.clone().to(torch.device('cuda:0'))
        xt_vis_list = []
        xt_vis_list.append(xt_temp)
        for pc_idx in range(2):
            vk = vT_list[pc_idx][0, :].view(-1, *xt.shape[1:])
            xt_edit = xt_temp + self.x_space_guidance_scale * self.x_space_guidance_num_step * vk
            xt_temp = xt_edit.clone().to(torch.device('cuda:0'))
            xt_vis_list.append(xt_edit)

        self.EXP_NAME = f'{idx}-Edit_xt-noise-{BASIS_NAME}'
        xt_vis = torch.cat(xt_vis_list, dim=0)
        self.DDIMforwardsteps(xt_vis, t_start_idx=self.edit_t_idx, t_end_idx=-1, performance_boosting=True)

        return xt
    

    @torch.no_grad()
    def run_edit_null_space_projection(
            self, idx, vis_num, vis_num_pc=5, pca_rank=50, pca_rank_null=10, op='mid', block_idx=0, null_space_projection = True, 
            encoder_decoder_by_et=False, use_mask = True, random_edit = False,
            **kwargs,
        ):
        '''
        1. xT -> xt -> ht -> x^0 -> x(t-1) -> x0 ; we edit xT based on Jacobian of x^0 on xT
        2. get local basis of h-space (u) and x-space (v) by approximating the SVD of the jacobian of f : xT -> x^0
        3. edit sample with predefined step size (sample += step_size * v)

        Args
            - idx               : sample idx
            - vis_num           : number of visualization editing steps
            - vis_num_pc        : number of visualization pc indexs
            - pca_rank          : pca rank
        '''
        # get latent code

        if self.dataset_name == 'Random':
            xT = torch.randn(1, self.c_in, self.image_size, self.image_size, dtype=self.dtype, device=self.device)
            self.EXP_NAME = "original"
            if (not os.path.exists(os.path.join(self.result_folder, "original.png"))) or (not os.path.exists(os.path.join(self.result_folder, "mask/mask.pt"))):
                print("Generating images and creating masks......")
                x0 = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=-1)
                x0_pil = Image.fromarray(((x0 / 2 + 0.5).clamp(0, 1)* 255).to(torch.uint8).permute(0, 2, 3, 1)[0].detach().cpu().numpy())
                masks = self.sam.mask_segmentation(x0_pil, resolution = self.image_size)
            else:
                print("Loading masks......")
                masks = torch.load(os.path.join(self.result_folder, "mask/mask.pt"))
            if self.args.sampling_mode:
                return None
            mask = masks[self.args.mask_index].squeeze(dim=0).repeat(3, 1, 1)
        elif self.dataset_name == "CelebA_HQ_mask":
            xT = self.run_DDIMinversion(idx=idx)

            mask = self.dataset.getmask(idx = self.args.sample_idx, choose_sem = self.args.choose_sem)
        elif self.dataset_name in ["FFHQ", "AFHQ", "Metface", "Flower", "LSUN_church", "LSUN_bedroom"]:
            if (not os.path.exists(os.path.join(self.result_folder, "original.png"))) or (not os.path.exists(os.path.join(self.result_folder, "mask/mask.pt"))):
                x0 = self.dataset[idx]
                tvu.save_image((x0 / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'original.png'))       
                x0_pil = Image.fromarray(((x0 / 2 + 0.5).clamp(0, 1)* 255).to(torch.uint8).permute(0, 2, 3, 1)[0].detach().cpu().numpy())
                masks = self.sam.mask_segmentation(x0_pil, resolution = self.image_size)
            else:
                print("Loading masks......")
                masks = torch.load(os.path.join(self.result_folder, "mask/mask.pt"))
            if self.args.sampling_mode:
                return None
            xT = self.run_DDIMinversion(idx=idx)
            if use_mask:
                mask = masks[self.args.mask_index].squeeze(dim=0).repeat(3, 1, 1)
            else:
                mask = None


        # xT -> xt
        xt, t, t_idx = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=self.edit_t_idx)

        assert t_idx == self.edit_t_idx

        if not os.path.exists(self.vT_path):
            print('!!!CALCULATING VT!!!')
            # get local basis
            if self.dataset_name == "CelebA_HQ_mask":
                save_dir = os.path.join(self.result_folder, "basis", f'local_basis-{self.edit_t}T-select-mask-{self.args.choose_sem}')
            elif self.dataset_name in ["FFHQ", "AFHQ", "Metface", "Flower", "LSUN_church", "LSUN_bedroom"]:
                save_dir = os.path.join(self.result_folder, "basis", f'local_basis-{self.edit_t}T-select-mask-{self.args.mask_index}')
            os.makedirs(save_dir, exist_ok=True)  
            vT_modify_path = os.path.join(save_dir, f'vT-modify-pca-rank-{pca_rank}.pt')
            vT_null_path = os.path.join(save_dir, f'vT-null-{pca_rank_null}.pt')   

            # load pre-computed local basis
            if os.path.exists(vT_modify_path):
                vT_modify = torch.load(vT_modify_path, map_location=self.device).type(self.dtype)
            # computed local basis using power method approximation
            else:
                print('!!!RUN LOCAL PULLBACK FOR EDIR SPACE!!!')
                xt = xt.to(device=self.device, dtype=self.dtype)

                u_modify, s_modify, vT_modify = self.local_encoder_decoder_pullback_xt(
                    x=xt, t=t, op=op, block_idx=block_idx, pca_rank=pca_rank, 
                    min_iter=10, max_iter=50, convergence_threshold=1e-4, mask = mask, noise=encoder_decoder_by_et
                )

                torch.save(vT_modify, vT_modify_path)

            if null_space_projection  and os.path.exists(vT_null_path):
                vT_null = torch.load(vT_null_path, map_location=self.device).type(self.dtype)
            elif not null_space_projection:
                pass
            else:
                print('!!!RUN LOCAL PULLBACK FOR NULL SPACE!!!')
                u_null, s_null, vT_null = self.local_encoder_decoder_pullback_xt(
                    x=xt, t=t, op=op, block_idx=block_idx, pca_rank=pca_rank_null, 
                    min_iter=10, max_iter=50, convergence_threshold=1e-4, mask = ~mask, noise=encoder_decoder_by_et
                )
                torch.save(vT_null, vT_null_path)

            if random_edit:
                vT_modify = torch.randn_like(vT_modify)

            # normalize vT
            if not null_space_projection:
                vT = vT_modify / vT_modify.norm(dim=1, keepdim=True)
            else:
                vT_null = vT_null[:pca_rank_null, :]
                vT = (vT_null.T @ (vT_null @ vT_modify.T)).T
                vT = vT_modify - vT
                vT = vT / vT.norm(dim=1, keepdim=True)
            if self.dataset_name == "CelebA_HQ_mask":
                BASIS_NAME = f"{encoder_decoder_by_et}_{self.args.choose_sem}-edit_{self.edit_t}T_null_proj_{null_space_projection}_rank{pca_rank_null}_scale_{self.x_space_guidance_scale}"
            elif self.dataset_name in ["FFHQ", "AFHQ", "Metface", "Flower", "LSUN_church", "LSUN_bedroom"]:
                BASIS_NAME = f"{encoder_decoder_by_et}_{self.args.mask_index}-edit_{self.edit_t}T_null_proj_{null_space_projection}_rank{pca_rank_null}_scale_{self.x_space_guidance_scale}"
            
            for pc_idx in range(max(vis_num_pc, vT.shape[0])):
                self.EXP_NAME = f'{idx}-Edit_xt-noise-{BASIS_NAME}-pc_{pc_idx:0=3d}'
                vT_path = os.path.join(save_dir, f'{self.EXP_NAME}-vT.pt')
                torch.save(vT[[pc_idx], :], vT_path)        
        else:
            print('!!!LOAD VT FROM VT_PATH!!!')
            vT = torch.load(self.vT_path)
            BASIS_NAME = f"edit_{self.edit_t}T-load-basis-'{os.path.basename(self.vT_path)}'"

        # edit
        original_xt = xt.detach()
        for pc_idx in range(min(vis_num_pc, vT.shape[0])):
            xts = {
                -1: None,
                1: None,
            }
            self.EXP_NAME = f'{idx}-Edit-random{random_edit}_xt-noise-{BASIS_NAME}-pc_{pc_idx:0=3d}'
            for direction in [1, -1]:
                # directly edit xt with vk
                vk = direction*vT[pc_idx, :].view(-1, *xt.shape[1:])

                xt_list = [original_xt.clone().to(torch.device('cuda:0'))]
                for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                    xt_edit = self.x_space_guidance_direct(
                        xt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                        single_edit_step=self.x_space_guidance_edit_step,
                    )
                    xt_list.append(xt_edit)
                xt = torch.cat(xt_list, dim=0)
                if vis_num == 1:
                    xt = xt[[0,-1],:]
                else:
                    xt = xt[::(xt.size(0) // vis_num)]
                xts[direction] = xt
            xt = torch.cat([(xts[-1].flip(dims=[0]))[:-1], xts[1]], dim=0)
            self.DDIMforwardsteps(xt, t_start_idx=self.edit_t_idx, t_end_idx=-1, performance_boosting=True)

        return xt


    def get_x0(self, t, x, mask = None):

        # x0 = self.scheduler.step(
        #         et, t, x, eta=0, use_clipped_model_output=None, generator=None
        #     ).x0

        et = self.unet(x, t)
        if not isinstance(et, torch.Tensor):
            et = et.sample

        t_next = self.scheduler.get_timesteps(t)

        # extract need parameters : at, at_next
        at = extract(self.scheduler.return_alphas_cumprod(), t, x.shape)
        at_next = extract(self.scheduler.return_alphas_cumprod(), t_next, x.shape)

        # DDIM step ; xt-1 = sqrt(at-1 / at) (xt - sqrt(1-at)*e(xt, t)) + sqrt(1-at-1)*e(xt, t)
        P_xt = (x - et * (1 - at).sqrt()) / at.sqrt()

        if mask is not None:
            # mask = mask.repeat(P_xt.shape[0], 1, 1, 1)
            P_xt = P_xt[:, mask]
        return P_xt


    def get_et(self, t, x, mask = None):

        et = self.unet(x, t)
        if not isinstance(et, torch.Tensor):
            et = et.sample

        if mask is not None:
            # mask = mask.repeat(P_xt.shape[0], 1, 1, 1)
            et = et[:, mask]
        return et


    def local_encoder_decoder_pullback_xt(
            self, x, t, op=None, block_idx=None,
            pca_rank=50, chunk_size=25, min_iter=10, max_iter=100, convergence_threshold=1e-3, mask = None, noise = False
        ):
        '''
        Args
            - x : zt
            - op : ['down', 'mid', 'up']
            - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
            - pooling : ['pixel-sum', 'channel-sum', 'single-channel', 'multiple-channel']
        Returns
            - h : hidden feature
        '''
        num_chunk = pca_rank // chunk_size if pca_rank % chunk_size == 0 else pca_rank // chunk_size + 1
        # print(pca_rank, chunk_size, num_chunk) # 2, 25, 1

        # get h samples
        time_s = time.time()

        c_i, w_i, h_i = x.size(1), x.size(2), x.size(3)
        # c_o, w_o, h_o = h_shape[1], h_shape[2], h_shape[3]
        if mask is None:
            c_o, w_o, h_o = c_i, w_i, h_i # output shape of x^0
        else:
            l_o = mask.sum().item()

        a = torch.tensor(0., device=x.device, dtype=x.dtype)

        # Algorithm 1
        vT = torch.randn(c_i*w_i*h_i, pca_rank, device=x.device, dtype=torch.float)
        vT, _ = torch.linalg.qr(vT)
        v = vT.T
        v = v.view(-1, c_i, w_i, h_i)


        time_s = time.time()
        # Jacobian subspace iteration
        for i in range(max_iter):
            v = v.to(device=x.device, dtype=x.dtype)
            v_prev = v.detach().cpu().clone()
            
            u = []
            v_buffer = list(v.chunk(num_chunk))
            for vi in v_buffer:
                if not noise:
                    g = lambda a : self.get_x0(t, x + a*vi, mask=mask)
                else:
                    g = lambda a : self.get_et(t, x + a*vi, mask=mask)
                
                ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a) # ui = J@vi
                u.append(ui.detach().cpu().clone())
            u = torch.cat(u, dim=0)
            u = u.to(x.device, x.dtype)

            if mask is None:
                if not noise:
                    g = lambda x : einsum(
                        u, self.get_x0(t, x, mask=mask), 'b c w h, i c w h -> b'
                    )
                else:
                    g = lambda x : einsum(
                        u, self.get_et(t, x, mask=mask), 'b c w h, i c w h -> b'
                    )
            else:
                if not noise:
                    g = lambda x : einsum(
                        u, self.get_x0(t, x, mask=mask), 'b l, i l -> b'
                    )     
                else:
                    g = lambda x : einsum(
                        u, self.get_et(t, x, mask=mask), 'b l, i l -> b'
                    )           
            
            v_ = torch.autograd.functional.jacobian(g, x) # vi = ui.T@J
            v_ = v_.view(-1, c_i*w_i*h_i)

            _, s, v = torch.linalg.svd(v_, full_matrices=False)
            v = v.view(-1, c_i, w_i, h_i)
            if mask is None:
                u = u.view(-1, c_o, w_o, h_o)
            else:
                u = u.view(-1, l_o)
            
            convergence = torch.dist(v_prev, v.detach().cpu()).item()
            print(f'power method : {i}-th step convergence : ', convergence)
            
            if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
                print('reach convergence threshold : ', convergence)
                break

        time_e = time.time()
        print('power method runtime ==', time_e - time_s)

        if mask is None:
            u, s, vT = u.reshape(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()
        else:
            u, s, vT = u.reshape(-1, l_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()

        return u, s, vT
    
    
    @torch.no_grad()
    def DDIMforwardsteps(
            self, xt, t_start_idx, t_end_idx, vis_psd=False, save_image=True, return_xt=True, performance_boosting=False,
        ):
        '''
        Args
            xt              : latent variable
            t_start_idx     : current timestep
            t_end_idx       : target timestep (t_end_idx = -1 for full forward)
            buffer_device   : device to store buffer
            vis_psd         : visualize psd
            save_image      : save image
            return_xt       : return xt
        '''
        print('start DDIMforward')
        # before start
        assert (t_start_idx < self.for_steps) & (t_end_idx <= self.for_steps)
        use_clipped_model_output = None
        num_inference_steps = self.for_steps
        eta = 0

        # set timestep (we do not use default scheduler set timestep method)
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        else:
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        # print(f'timesteps : {timesteps}')

        # save traj
        xt_traj = [xt.clone()]
        et_traj = []

        ##################
        # denoising loop #
        ##################
        for i, t in enumerate(timesteps):
            # skip timestpe if not in range
            if t_end_idx == i:
                print('t_end_idx : ', i)
                return xt, t, i
        
            elif (i < t_start_idx): 
                continue

            elif t_start_idx == i:
                print('t_start_idx : ', i)

            if performance_boosting & (self.performance_boosting_t_idx <= i) & (self.performance_boosting_t_idx != len(timesteps)-1):
                eta = 1
            else:
                eta = 0

            # use buffer to avoid OOM
            xt = xt.to(device=self.buffer_device)
            if xt.size(0) // self.memory_bound == 0:
                xt_buffer = [xt]
            else:
                xt_buffer = list(xt.chunk(xt.size(0) // self.memory_bound))

            for buffer_idx, xt in enumerate(xt_buffer):
                xt = xt.to(device=self.device, dtype=self.dtype)

                # 1. predict noise model_output
                et = self.unet(xt, t)
                if not isinstance(et, torch.Tensor):
                    et = et.sample

                # 2. predict previous mean of xt x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                xt = self.scheduler.step(
                    et, t, xt, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=None
                ).prev_sample

                # save xt in buffer
                xt_buffer[buffer_idx] = xt.to(self.buffer_device)

            # save traj
            if buffer_idx == 0:
                xt_traj.append(xt[0].detach().cpu().type(torch.float32).clone())
                et_traj.append(et[0].detach().cpu().type(torch.float32).clone())

            xt = torch.cat(xt_buffer, dim=0)
            del xt_buffer
            torch.cuda.empty_cache()

        if save_image:
            image = (xt / 2 + 0.5).clamp(0, 1)
            tvu.save_image(
                image, os.path.join(self.result_folder, f'{self.EXP_NAME}.png'), nrow = image.size(0),
            )
            
        # visualize power spectral density of zt_traj, et_traj
        if vis_psd:
            vis_power_spectral_density(
                et_traj, save_path=os.path.join(self.obs_folder, f'et_psd-{self.EXP_NAME}.png')
            )

            vis_power_spectral_density(
                xt_traj, save_path=os.path.join(self.obs_folder, f'xt_psd-{self.EXP_NAME}.png')
            )

        if return_xt:
            return xt

        return


    @torch.no_grad()
    def x_space_guidance_direct(self, xt, t_idx, vk, single_edit_step):
        # necesary parameters
        t = self.scheduler.timesteps[t_idx]

        # edit xt with vk
        xt_edit = xt + self.x_space_guidance_scale * single_edit_step * vk

        return xt_edit


    
####################
# Custom timesteps #
####################
from functools import partial
from typing import Union

def custom_set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, inversion_flag: bool = False):
    """
    Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
    Args:
        num_inference_steps (`int`):
            the number of diffusion steps used when generating samples with a pre-trained model.
    """

    if num_inference_steps > self.config.num_train_timesteps:
        raise ValueError(
            f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
            f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
            f" maximal {self.config.num_train_timesteps} timesteps."
        )

    self.num_inference_steps = num_inference_steps
    step_ratio = self.config.num_train_timesteps // self.num_inference_steps
    # creates integer timesteps by multiplying by ratio
    # casting to int to avoid issues when num_inference_step is power of 3
    # timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().copy().astype(np.int64)
    timesteps = np.linspace(0, 1, num_inference_steps) * (self.config.num_train_timesteps-2) # T=999
    timesteps = timesteps + 1e-6
    timesteps = timesteps.round().astype(np.int64)
    # reverse timesteps except inverse diffusion
    # keep it numpy array
    if not inversion_flag:
        timesteps = np.flip(timesteps).copy()

    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.timesteps += self.config.steps_offset