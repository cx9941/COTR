set -o errexit
for seed in 0
do
export CUDA_VISIBLE_DEVICES=4,5,6,7
for turn in 0 1 2 3 4 5 6 7 8 9 10 11 12
do
accelerate launch --num_processes=4 --main_process_port=7888 1.llm_select.py --turn $turn --seed $seed
python 2.result_clean.py  --turn $turn
done
done
