# this codebase is built on https://github.com/enkeejunior1/Diffusion-Pullback 

from utils.define_argparser import parse_args, preset

from modules.edit import (
    EditStableDiffusion,
    EditUncondDiffusion,
    EditDeepFloydIF,
    EditLatentConsistency
)

if __name__ == "__main__":
    ##########
    # preset #
    ##########
    # parse args
    args = parse_args()
    
    # preset
    args = preset(args)

    # get instance
    if args.is_stable_diffusion:
        print('is stable-diffusion')
        edit = EditStableDiffusion(args)
    elif args.is_DeepFloyd_IF_diffusion:
        print('is DeepFloyd-IF')
        edit = EditDeepFloydIF(args) 
    elif args.is_LCM:
        print('is LCM')
        edit = EditLatentConsistency(args)       
    else:
        print('is custmized diffusion model')
        edit = EditUncondDiffusion(args)
    
    ########################################
    # experiment : local direction editing #
    ########################################
    if args.run_edit_local_encoder_decoder_pullback_zt:
        edit.run_edit_local_encoder_decoder_pullback_zt(
            idx=args.sample_idx, op='mid', block_idx=0,
            vis_num=4, vis_num_pc=2, pca_rank=2, edit_prompt=args.edit_prompt,
            encoder_decoder_by_et = args.encoder_decoder_by_et
        )

    # DDPM
    if args.run_edit_null_space_projection:
        edit.run_edit_null_space_projection(
            idx=args.sample_idx, op='mid', block_idx=0,
            vis_num=args.vis_num, vis_num_pc=args.pca_rank, pca_rank=args.pca_rank, edit_prompt=args.edit_prompt, null_space_projection = args.null_space_projection, pca_rank_null=args.pca_rank_null, 
            encoder_decoder_by_et = args.encoder_decoder_by_et, use_mask=args.use_mask, random_edit = args.random_edit
        )
   
    # T2I
    if args.run_edit_null_space_projection_zt:
        edit.run_edit_null_space_projection_zt(
            op='mid', block_idx=0, mask_index = args.mask_index,
            vis_num=args.vis_num, vis_num_pc=args.pca_rank, pca_rank=args.pca_rank, 
            edit_prompt=args.edit_prompt, 
            null_space_projection = args.null_space_projection, pca_rank_null=args.pca_rank_null, 
            non_semantic = args.non_semantic
        )
    if args.run_edit_null_space_projection_zt_semantic:
        edit.run_edit_null_space_projection_zt_semantic(
            op='mid', block_idx=0, mask_index = args.mask_index,
            vis_num=args.vis_num, vis_num_pc=args.pca_rank, pca_rank=args.pca_rank, 
            edit_prompt=args.edit_prompt, 
            null_space_projection = args.null_space_projection, pca_rank_null=args.pca_rank_null, 
        )
    if args.run_edit_null_space_projection_xt:
        edit.run_edit_null_space_projection_xt(
            op='mid', block_idx=0, mask_index = args.mask_index,
            vis_num=args.vis_num, vis_num_pc=args.pca_rank, pca_rank=args.pca_rank, 
            edit_prompt=args.edit_prompt, 
            null_space_projection = args.null_space_projection, pca_rank_null=args.pca_rank_null, 
        )        
    if args.run_edit_null_space_projection_xt_semantic:
        edit.run_edit_null_space_projection_xt_semantic(
            op='mid', block_idx=0, mask_index = args.mask_index,
            vis_num=args.vis_num, vis_num_pc=args.pca_rank, pca_rank=args.pca_rank, 
            edit_prompt=args.edit_prompt, 
            null_space_projection = args.null_space_projection, pca_rank_null=args.pca_rank_null, 
            jacobian=args.jacobian
        )     

    # others
    if args.group_edit_null_space_projection:
        edit.group_edit_null_space_projection(
            idx=args.sample_idx, op='mid', block_idx=0, vis_num_pc=1, pca_rank=1, edit_prompt=args.edit_prompt, null_space_projection = args.null_space_projection, pca_rank_null=args.pca_rank_null, 
            encoder_decoder_by_et = args.encoder_decoder_by_et
        )


    #####################
    # simple experiment #
    #####################
    # experiment : forward (for debug diffusion model load)
    if args.run_ddim_forward:
        edit.run_DDIMforward(num_samples=5)
    
    # experiment : inversion
    if args.run_ddim_inversion:
        edit.run_DDIMinversion(idx=args.sample_idx)
