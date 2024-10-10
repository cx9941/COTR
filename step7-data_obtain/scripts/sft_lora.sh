set -o errexit
# export CUDA_VISIBLE_DEVICES='5,6,7'
for dataset_name in 'en' 'eu' 'jp'
do
for model_name in 'Baichuan2-7B-Base' 'Baichuan2-13B-Base' 'Llama-2-13b-hf'
do
DS_SKIP_CUDA_CHECK=1 deepspeed --include localhost:1,2 lora_prompt_embedding.py \
    --model_dir /home/data/qinchuan/TMIS/llm_models/${model_name} \
    --output_dir /home/data/qinchuan/TMIS/paper_code/output/data_obtain/lora/${model_name} \
    --data_path /home/data/qinchuan/TMIS/paper_code/task_data/${dataset_name}/${dataset_name}-gpt.csv \
    --rank_path /home/data/qinchuan/TMIS/paper_code/task_data/${dataset_name}/rank_bert.json \
    --task_name $dataset_name
done
done

# DS_SKIP_CUDA_CHECK=1 deepspeed --include localhost:1,2,3,4 lora_prompt_embedding.py --output_dir {} --data_path {} --rank_path {} --label_path {标签体系位置，看下默认值就知道了，随数据集变化} --model_name {模型简写，可以看choice} --task_name {数据集名称}
