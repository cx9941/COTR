set -o errexit
for dataset_name in 'en-enhanced'
do
for model_name in 'llama' 'baichuan' 'nanbeige'
do
deepspeed --include localhost:0,1 sft.py \
    --model_name $model_name \
    --output_dir outputs/${dataset_name}/${model_name}/sft \
    --data_path data/${dataset_name}/${dataset_name}-gpt.csv \
    --rank_path data/${dataset_name}/rank_bert.json \
    --task_name $dataset_name
done
done