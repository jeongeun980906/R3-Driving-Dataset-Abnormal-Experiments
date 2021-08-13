cd ..
python main.py --gpu 1 --id 2 --init_pool 1000 --query_size 500 --query_step 20 --epoch 200
python main.py --query_method aleatoric --gpu 1 --id 2 --init_pool 1000 --query_size 500 --query_step 20 --epoch 200
python main.py --query_method pi_entropy --gpu 1 --id 2 --init_pool 1000 --query_size 500 --query_step 20 --epoch 200
python main.py --query_method random --gpu 1 --id 2 --init_pool 1000 --query_size 500 --query_step 20 --epoch 200