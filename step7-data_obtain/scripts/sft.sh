set -o errexit
# export CUDA_VISIBLE_DEVICES='5,6,7'
for dataset_name in 'en' 'eu' 'jp'
do
for model_name in 'Meta-Llama-3.1-8B-Instruct' 'Baichuan2-7B-Chat' 'Nanbeige2-8B-Chat'
do
deepspeed --include localhost:0,7 sft.py \
    --model_dir /home/data/qinchuan/TMIS/llm_models/${model_name} \
    --output_dir /home/data/qinchuan/TMIS/paper_code/output/data_obtain/sft/${dataset_name}/${model_name} \
    --data_path /home/data/qinchuan/TMIS/paper_code/task_data/${dataset_name}/${dataset_name}-gpt.csv \
    --rank_path /home/data/qinchuan/TMIS/paper_code/task_data/${dataset_name}/rank_bert.json
done
done