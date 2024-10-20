DATASET_PATHS = {
	# uncond diffusion model
	'CelebA_HQ'	: 'datasets/celeba_hq',
	'AFHQ'     	: 'datasets/afhq',
	'FFHQ'     	: 'datasets/ffhq',
	'LSUN_bedroom'	: 'datasets/lsun-bedroom',
	'LSUN_church'	: 'datasets/lsun-church',
	
	# stable diffusion 
    'Examples'  	: 'datasets/examples',
}

MODEL_PATHS = {
    'LSUN_bedroom'  : 'weights/lsun_bedroom.pt',
    'LSUN_cat'	    : 'weights/lsun_cat.pt',
    'LSUN_horse'    : 'weights/lsun_horse.pt',
    'AFHQ_P2'       : '/scratch/qingqu_root/qingqu1/shared_data/P2_diffusionmodel/afhqdog_p2.pt',
    'Flower_P2'     : '/scratch/qingqu_root/qingqu1/shared_data/P2_diffusionmodel/flower_p2.pt',
    'FFHQ_P2'       : '/scratch/qingqu_root/qingqu1/shared_data/P2_diffusionmodel/ffhq_p2.pt',
    'Cub_P2'        : '/scratch/qingqu_root/qingqu1/shared_data/P2_diffusionmodel/cub_p2.pt',
    'Metface_P2'    : '/scratch/qingqu_root/qingqu1/shared_data/P2_diffusionmodel/metface_p2.pt',
}