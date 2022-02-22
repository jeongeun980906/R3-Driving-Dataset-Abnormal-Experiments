cd ..
python main.py --frame 1 --id 1
python main.py --frame 5 --id 2
for i in {1..2}
do
    python plot.py --id $i
done

python main.py --frame 1 --id 1 --mode vae --h_dim 128 --z_dim 32
python main.py --frame 5 --id 2 --mode vae --h_dim 256 --z_dim 128

for i in {1..2}
do
    python plot.py --id $i --mode vae
done