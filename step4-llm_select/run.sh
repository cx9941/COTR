set -o errexit
for seed in 0 1 2 3 4
do
export CUDA_VISIBLE_DEVICES="0,7"
for turn in 0 1 2 3 4
do
for dataset_name in 'eu' 'en'
do
accelerate launch --num_processes=2 --main_process_port=7889 1.llm_select.py --turn $turn --seed $seed --dataset_name $dataset_name
python 2.result_clean.py  --turn $turn --dataset_name $dataset_name --seed $seed
done
done
done
