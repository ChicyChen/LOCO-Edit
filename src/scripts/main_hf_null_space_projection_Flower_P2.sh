for sample_idx in <TODO>
    do
    python main.py \
        --sh_file_name                          main_hf_null_space_projection_Flower_P2.sh    \
        --device                                cuda:0                                      \
        --dtype                                 fp32                                        \
        --seed                                  0                                           \
        --sample_idx                            $sample_idx                                         \
        --model_name                            Flower_P2                                     \
        --dataset_name                          Flower                                        \
        --mask_model_name                       facebook/sam-vit-large                      \
        --mask_type                             SAM                                         \
        --for_steps                             100                                         \
        --inv_steps                             100                                         \
        --use_yh_custom_scheduler               True                                        \
        --x_space_guidance_edit_step            1                                           \
        --x_space_guidance_scale                0.5                                         \
        --x_space_guidance_direct               True                                        \
        --x_space_guidance_num_step             16                                          \
        --edit_t                                0.7                                         \
        --performance_boosting_t                0.2                                         \
        --run_edit_null_space_projection        True                                        \
        --note                                  "Uncond"                                    \
        --null_space_projection                 True                                        \
        --pca_rank_null                         5                                           \
        --pca_rank                              5                                           \
        --mask_index                            <TODO>                                           \
        --sampling_mode                         True                                       \
        --vis_num                               2                                           
    done

        # --vT_path                               "*.pt"