cd ..
python main.py --frame 1 --id 11 --mode vae --gpu 1 --h_dim 32 --z_dim 16
python main.py --frame 5 --id 12 --mode vae --gpu 1 --h_dim 128 --z_dim 64
python main.py --frame 10  --id 13 --mode vae --gpu 1 --h_dim 128 --z_dim 64

python main.py --frame 1 --id 14 --exp_case 2 3 --mode vae --gpu 1 --h_dim 32 --z_dim 16
python main.py --frame 5 --id 15 --exp_case 2 3 --mode vae --gpu 1 --h_dim 128 --z_dim 64
python main.py --frame 10  --id 16 --exp_case 2 3 --mode vae --gpu 1 --h_dim 128 --z_dim 64
 
python main.py --frame 1 --id 17 --exp_case 1 --mode vae --gpu 1 --h_dim 32 --z_dim 16
python main.py --frame 5 --id 18 --exp_case 1 --mode vae --gpu 1 --h_dim 128 --z_dim 64
python main.py --frame 10  --id 19 --exp_case 1 --mode vae --gpu 1 --h_dim 128 --z_dim 64