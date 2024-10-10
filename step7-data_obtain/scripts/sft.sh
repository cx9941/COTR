set -o errexit
for dataset_name in 'en' 'eu' 'jp'
do
for model_name in 'Baichuan2-7B-Chat' 'Nanbeige2-8B-Chat' 'Meta-Llama-3.1-8B'
do
deepspeed --include localhost:1,2 sft.py \
    --model_dir ../llms/${model_name} \
    --output_dir outputs/${dataset_name}/${model_name} \
    --data_path data/${dataset_name}/${dataset_name}-gpt.csv \
    --rank_path data/${dataset_name}/rank_bert.json
done
done