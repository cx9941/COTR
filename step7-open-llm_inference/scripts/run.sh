set -o errexit
export CUDA_VISIBLE_DEVICES="2,4"
for seed in 0
do
for model_name in 'baichuan' 'nanbeige' 'llama'
do
for dataset_name in 'en' 'eu' 'jp'
do
accelerate launch --num_processes=2 --main_process_port=7889  1.llm_select.py --model_name $model_name --dataset_name $dataset_name --seed $seed
done
done
done