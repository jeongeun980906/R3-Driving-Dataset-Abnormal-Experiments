cd ..
python main.py --frame 1 --id 21 --mode vae --h_dim 128 --z_dim 32
python main.py --frame 5 --id 22 --mode vae --h_dim 256 --z_dim 128
python main.py --frame 10  --id 23 --mode vae --h_dim 512 --z_dim 256

python main.py --frame 1 --id 24 --exp_case 2 3 --mode vae --h_dim 128 --z_dim 32
python main.py --frame 5 --id 25 --exp_case 2 3 --mode vae ---h_dim 256 --z_dim 128
python main.py --frame 10  --id 26 --exp_case 2 3 --mode vae --h_dim 512 --z_dim 256

python main.py --frame 1 --id 27 --exp_case 1 --mode vae --h_dim 128 --z_dim 32
python main.py --frame 5 --id 28 --exp_case 1 --mode vae --h_dim 256 --z_dim 128
python main.py --frame 10  --id 29 --exp_case 1 --mode vae --h_dim 512 --z_dim 256
for i in {21..29}
do
    python plot.py --id $i --mode vae
done