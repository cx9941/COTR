for seed in 2 
do
accelerate launch --num_processes=2 1.py --seed $seed
done