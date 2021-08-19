cd ..
python main.py --gpu 0 --id 5 --exp_case 2 --init_pool 100 --query_size 50
python main.py --query_method aleatoric --gpu 1 --id 5 --exp_case 2 --init_pool 100 --query_size 50
python main.py --query_method pi_entropy --gpu 1 --id 5 --exp_case 2 --init_pool 100 --query_size 50
python main.py --query_method random --gpu 1 --id 5 --exp_case 2 --init_pool 100 --query_size 50

python main.py --gpu 0 --id 6 --exp_case 3 --init_pool 100 --query_size 50
python main.py --query_method aleatoric --gpu 1 --id 6 --exp_case 3 --init_pool 100 --query_size 50
python main.py --query_method pi_entropy --gpu 1 --id 6 --exp_case 3 --init_pool 100 --query_size 50
python main.py --query_method random --gpu 1 --id 6 --exp_case 3 --init_pool 100 --query_size 50