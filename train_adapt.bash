python train.py --mode adapt --name ade20k_adapt_soda_nce --dataroot ../mmsegmentation/data --dataset_mode adaptation --output_nc 21 --n_epochs 100 --batch_size 8 --lr 0.01 --crop_size 480 --save_epoch_freq 1 --display_freq 100 --display_env ade20k_adapt_soda_nce --lambda_GAN 0.0 --lambda_NCE 0.0
