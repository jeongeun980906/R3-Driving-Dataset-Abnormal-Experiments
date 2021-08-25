cd ..
python main.py --gpu 1 --init_pool 1000 --query_size 500 --id 2 --exp_case 2 3 
python main.py --gpu 1 --query_method aleatoric --init_pool 1000 --query_size 500 --id 2 --exp_case 2 3 
python main.py --gpu 1 --query_method pi_entropy --init_pool 1000 --query_size 500 --id 2 --exp_case 2 3 
python main.py --gpu 1 --query_method random --init_pool 1000 --query_size 500 --id 2 --exp_case 2 3 
# python main.py --gpu 0 --id 4 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method aleatoric --gpu 0 --id 4 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method pi_entropy --gpu 0 --id 4 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method random --gpu 0 --id 4 --exp_case 1 --init_pool 100 --query_size 50

# python main.py --gpu 0 --id 4 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method aleatoric --gpu 0 --id 4 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method pi_entropy --gpu 0 --id 4 --exp_case 1 --init_pool 100 --query_size 50
# python main.py --query_method random --gpu 0 --id 4 --exp_case 1 --init_pool 100 --query_size 50

# python main.py --gpu 1 --id 4 --init_pool 200 --query_size 100 
# python main.py --query_method aleatoric --gpu 1 --id 4 --init_pool 200 --query_size 100
# python main.py --query_method pi_entropy --gpu 1 --id 4 --init_pool 200 --query_size 100
# python main.py --query_method random --gpu 1 --id 4 --init_pool 200 --query_size 100