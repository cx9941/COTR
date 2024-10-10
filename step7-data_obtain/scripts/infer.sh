set -o errexit
export CUDA_VISIBLE_DEVICES='4'
for dataset_name in 'en' 'eu' 'jp'
do
for model_name in 'Llama-2-13b-hf'
do
python infer.py --lora_true false  \
    --model_dir /home/data/qinchuan/TMIS/llm_models/${model_name} \
    --output_dir /home/data/qinchuan/TMIS/paper_code/output/data_obtain/infer_nosft/${dataset_name}/${model_name} \
    --data_path /home/data/qinchuan/TMIS/paper_code/task_data/${dataset_name}/${dataset_name}-gpt.csv \
    --rank_path /home/data/qinchuan/TMIS/paper_code/task_data/${dataset_name}/rank_bert.json
done
done