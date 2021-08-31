cd ..
python main.py --frame 1 --id 1 --mode vae --gpu 1
python main.py --frame 5 --id 2 --mode vae --gpu 1 --h_dim 100 --h_dim 64
python main.py --frame 10  --id 3 --mode vae --gpu 1 --h_dim 100 --h_dim 64

python main.py --frame 1 --id 4 --exp_case 2 3 --mode vae --gpu 1
python main.py --frame 5 --id 5 --exp_case 2 3 --mode vae --gpu 1 --h_dim 100 --h_dim 64
python main.py --frame 10  --id 6 --exp_case 2 3 --mode vae --gpu 1 --h_dim 100 --h_dim 64

python main.py --frame 1 --id 8 --exp_case 1 --mode vae --gpu 1
python main.py --frame 5 --id 9 --exp_case 1 --mode vae --gpu 1 --h_dim 100 --h_dim 64
python main.py --frame 10  --id 10 --exp_case 1 --mode vae --gpu 1 --h_dim 100 --h_dim 64