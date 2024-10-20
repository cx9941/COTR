export CUDA_VISIBLE_DEVICES="0,1"
for seed in 0
do
for model_name in 'baichuan-sft-enhanced' 'nanbeige-sft-enhanced' 'llama-sft-enhanced'
do
for dataset_name in 'en'
do
accelerate launch --num_processes=2 --main_process_port=7889  1.llm_select.py --model_name $model_name --dataset_name $dataset_name --seed $seed --mode normal
for thred in 50
do
python 2.result_get.py --model_name $model_name --dataset_name $dataset_name --seed $seed --thred $thred --mode normal
done
done
done
done