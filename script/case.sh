cd ..
python main.py --gpu 0 --init_pool 100 --query_size 300 --id 11 --n_object 1
python main.py --gpu 0 --query_method aleatoric --init_pool 100 --query_size 300 --id 11 --n_object 1
python main.py --gpu 0 --query_method pi_entropy --init_pool 100 --query_size 300 --id 11 --n_object 1
python main.py --gpu 0 --query_method random --init_pool 100 --query_size 300 --id 11 --n_object 1
# python main.py --gpu 0 --id 5 --exp_case 2 --init_pool 800 --query_size 400
# python main.py --query_method aleatoric --gpu 0 --id 5 --exp_case 2 --init_pool 800 --query_size 400
# python main.py --query_method pi_entropy --gpu 0 --id 5 --exp_case 2 --init_pool 800 --query_size 400
# python main.py --query_method random --gpu 0 --id 5 --exp_case 2 --init_pool 800 --query_size 400

# python main.py --gpu 0 --id 5 --n_object 1 --exp_case 3 --init_pool 800 --query_size 400
# python main.py --query_method aleatoric --gpu 0 --id 5 --n_object 1 --exp_case 3 --init_pool 800 --query_size 400
# python main.py --query_method pi_entropy --gpu 0 --id 5 --n_object 1 --exp_case 3 --init_pool 800 --query_size 400
# python main.py --query_method random --gpu 0 --id 5 --n_object 1 --exp_case 3 --init_pool 800 --query_size 400