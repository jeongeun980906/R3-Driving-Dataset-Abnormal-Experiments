cd ..
python main.py --gpu 1 --id 1 --exp_case 1
python main.py --query_method aleatoric --gpu 1 --id 1 --exp_case 1
python main.py --query_method pi_entropy --gpu 1 --id 1 --exp_case 1
python main.py --query_method random --gpu 1 --id 1 --exp_case 1

# python main.py --gpu 1 --id 4 --init_pool 200 --query_size 100 
# python main.py --query_method aleatoric --gpu 1 --id 4 --init_pool 200 --query_size 100
# python main.py --query_method pi_entropy --gpu 1 --id 4 --init_pool 200 --query_size 100
# python main.py --query_method random --gpu 1 --id 4 --init_pool 200 --query_size 100