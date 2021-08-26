cd ..
python main.py --gpu 1 --init_pool 500 --query_size 250 --exp_case 2 3 --id 8 --n_object 1
python main.py --gpu 1 --query_method aleatoric --init_pool 500 --query_size 250 --exp_case 2 3 --id 8 --n_object 1
python main.py --gpu 1 --query_method pi_entropy --init_pool 500 --query_size 250 --exp_case 2 3 --id 8 --n_object 1
python main.py --gpu 1 --query_method random --init_pool 500 --query_size 250 --exp_case 2 3 --id 8 --n_object 1 
# python main.py --gpu 0 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method aleatoric --gpu 0 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method pi_entropy --gpu 0 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method random --gpu 0 --exp_case 1 --init_pool 100 --query_size 50

# python main.py --gpu 0 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method aleatoric --gpu 0 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method pi_entropy --gpu 0 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method random --gpu 0 --exp_case 1 --init_pool 100 --query_size 50

# python main.py --gpu 1 --init_pool 500 --query_size 250 
# python main.py --query_method aleatoric --gpu 1 --init_pool 500 --query_size 250
# python main.py --query_method pi_entropy --gpu 1 --init_pool 500 --query_size 250
# python main.py --query_method random --gpu 1 --init_pool 500 --query_size 250