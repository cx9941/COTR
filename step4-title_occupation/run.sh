set -o errexit
for seed in 0
do
export CUDA_VISIBLE_DEVICES=4,5,6,7
for turn in 0 1 2 3
do
for dataset_name in 'eu'
do
accelerate launch --num_processes=4 --main_process_port=7888 1.llm_select.py --turn $turn --seed $seed --dataset_name $dataset_name
python 2.result_clean.py  --turn $turn --dataset_name $dataset_name
done
done
done
