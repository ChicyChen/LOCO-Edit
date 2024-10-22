python main.py \
    --sh_file_name                          main_T2I_StableDiffusion_null_space_projection.sh           \
    --device                                cuda:0                                      \
    --model_name                            stabilityai/stable-diffusion-2-1-base                      \
    --mask_model_name                       facebook/sam-vit-large                      \
    --dataset_name                          Random                                      \
    --edit_prompt                           "a photo of a man wearing glasses"                  \
    --for_prompt                            "a photo of a man"                          \
    --neg_prompt                            ""                                          \
    --x_space_guidance_scale                0.2                                         \
    --x_space_guidance_num_step             16                                          \
    --edit_t                                0.7                                         \
    --run_edit_null_space_projection_zt     True                               \
    --note                                  "with_prompt"                               \
    --guidance_scale                        7.5                                         \
    --guidance_scale_edit                   4.0                                         \
    --seed                                  0                                 \
    --null_space_projection                 True                                        \
    --pca_rank_null                         5                                           \
    --pca_rank                              5                                           \
    --sampling_mode                         True                                       \
    --mask_index                            1                                          \
    --tilda_v_score_type                    "null+(for-null)+(edit-null)"               \
    --dtype                                 fp32                                        \
    --cache_folder                          <TODO>         \
    --vis_num                               2                                           \
    --use_sega                              False