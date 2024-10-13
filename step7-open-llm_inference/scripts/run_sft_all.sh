set -o errexit
export CUDA_VISIBLE_DEVICES="4,5"
for seed in 0
do
for model_name in 'baichuan-sft' 'nanbeige-sft' 'llama-sft' 'baichuan' 'nanbeige' 'llama'
do
for dataset_name in 'eu' 'en' 'jp'
do
accelerate launch --num_processes=2 --main_process_port=7889  1.llm_select.py --model_name $model_name --dataset_name $dataset_name --seed $seed --mode all
for thred in 50
do
python 2.result_get.py --model_name $model_name --dataset_name $dataset_name --seed $seed --thred $thred  --mode all
done
done
done
done