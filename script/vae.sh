cd ..
python main.py --frame 1 --id 11 --mode vae --h_dim 128 --z_dim 32
python main.py --frame 5 --id 12 --mode vae --h_dim 256 --z_dim 128
python main.py --frame 10  --id 13 --mode vae --h_dim 512 --z_dim 256

python main.py --frame 1 --id 14 --exp_case 2 3 --mode vae --h_dim 128 --z_dim 32
python main.py --frame 5 --id 15 --exp_case 2 3 --mode vae --h_dim 256 --z_dim 128
python main.py --frame 10  --id 16 --exp_case 2 3 --mode vae --h_dim 512 --z_dim 256

python main.py --frame 1 --id 17 --exp_case 1 --mode vae --h_dim 128 --z_dim 32
python main.py --frame 5 --id 18 --exp_case 1 --mode vae --h_dim 256 --z_dim 128
python main.py --frame 10  --id 19 --exp_case 1 --mode vae --h_dim 512 --z_dim 256
for i in {11..19}
do
    python plot.py --id $i --mode vae
done