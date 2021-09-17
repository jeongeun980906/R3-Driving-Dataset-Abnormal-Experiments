cd ..
python main.py --frame 1 --id 1
python main.py --frame 5 --id 2
for i in {1..2}
do
    python plot.py --id $i
done