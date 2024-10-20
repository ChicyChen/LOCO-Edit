for sample_idx in 4729
    do
    python main.py \
        --sh_file_name                          main_celeba_hf_null_space_projection.sh    \
        --sample_idx                            $sample_idx                                 \
        --device                                cuda:0                                      \
        --dtype                                 fp32                                        \
        --seed                                  0                                           \
        --model_name                            CelebA_HQ_HF                                \
        --dataset_name                          CelebA_HQ_mask                              \
        --for_steps                             100                                         \
        --inv_steps                             100                                         \
        --use_yh_custom_scheduler               True                                        \
        --x_space_guidance_edit_step            1                                           \
        --x_space_guidance_scale                0.5                                         \
        --x_space_guidance_num_step             16                                          \
        --edit_t                                0.6                                         \
        --performance_boosting_t                0.2                                         \
        --run_edit_null_space_projection        True                                        \
        --dataset_root                          "/scratch/qingqu_root/qingqu1/shared_data/celebA-HQ-mask/CelebAMask-HQ" \
        --choose_sem                            "l_eye"                                     \
        --null_space_projection                 True                                        \
        --use_mask                              True                                       \
        --pca_rank_null                         5                                           \
        --pca_rank                              1                                           \
        --vis_num                               2                                           
    done

# --vT_path                               "*.pt"

# examples in the form of (sample_idx, choose_sem):
# (3456, "hair"), (4729, "l_eye") (2984, "hair") (3638, "l_eye")