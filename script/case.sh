cd ..
python main.py --gpu 0 --exp_case 2 3
python main.py --gpu 0 --exp_case 2 3 --query_method aleatoric
python main.py --gpu 0 --exp_case 2 3 --query_method pi_entropy
python main.py --gpu 0 --exp_case 2 3 --query_method random
# python main.py --gpu 0 --id 5 --exp_case 2 --init_pool 800 --query_size 400
# python main.py --query_method aleatoric --gpu 1 --id 5 --exp_case 2 --init_pool 800 --query_size 400
# python main.py --query_method pi_entropy --gpu 1 --id 5 --exp_case 2 --init_pool 800 --query_size 400
# python main.py --query_method random --gpu 1 --id 5 --exp_case 2 --init_pool 800 --query_size 400

# python main.py --gpu 0 --id 4 --exp_case 3 --init_pool 800 --query_size 400
# python main.py --query_method aleatoric --gpu 1 --id 4 --exp_case 3 --init_pool 800 --query_size 400
# python main.py --query_method pi_entropy --gpu 1 --id 4 --exp_case 3 --init_pool 800 --query_size 400
# python main.py --query_method random --gpu 1 --id 4 --exp_case 3 --init_pool 800 --query_size 400