# 纯infer效果
python infer.py --lora_true false  --model_dir /home/data/qinchuan/TMIS/paper_code/llm_models/baichuan-inc/Baichuan2-7B-Chat --output_dir /home/data/qinchuan/TMIS/paper_code/output/data_obtain/baichuan/infer_nosft --data_path /home/data/qinchuan/TMIS/paper_code/task_data/eu/eu-gpt.csv --rank_path /home/data/qinchuan/TMIS/paper_code/task_data/eu/rank_bert.json --device_id 4
# 直接sft 效果, 注意model如果是baichuan，peft_config用W_pack,否则用q_proj, v_proj，local_host后面是4张卡序号
先训练：deepspeed --include localhost:4,5,6,7 sft.py --model_dir {} --output_dir {} --data_path {} --rank_path {}
推理：python infer.py --model_dir {} --output_dir {} --data_path {} --rank_path {} --adapter_dir {训练的output_dir中任意一个epoch，可以都试试} --device_id 4

# lora_prompt_embedding嵌入方法
先训练：DS_SKIP_CUDA_CHECK=1 deepspeed --include localhost:1,2,3,4 lora_prompt_embedding.py --output_dir {} --data_path {} --rank_path {} --label_path {标签体系位置，看下默认值就知道了，随数据集变化} --model_name {模型简写，可以看choice} --task_name {数据集名称}
推理：python lora_prompt_embedding.py --train_or_test test --model_dir {} --output_dir {} --data_path {} --rank_path {} --checkpoint {训练的output_dir中任意一个epoch，可以都试试} --device_id 4 --model_name {模型简写，可以看choice} --task_name {数据集名称}

# lora_prompt_embedding嵌入方法,不初始化init
先训练：DS_SKIP_CUDA_CHECK=1 deepspeed --include localhost:1,2,3,4 lora_prompt_embedding.py --output_dir {} --data_path {} --rank_path {} --label_path {标签体系位置，看下默认值就知道了，随数据集变化} --model_name {模型简写，可以看choice} --task_name {数据集名称} --init false
推理：python lora_prompt_embedding.py --train_or_test test --model_dir {} --output_dir {} --data_path {} --rank_path {} --checkpoint {训练的output_dir中任意一个epoch，可以都试试} --device_id 4 --model_name {模型简写，可以看choice} --task_name {数据集名称} --init false

# 测过测试
python result_gpt2.py --data_path {} --label_path {} --result_path {infer文件位置} --mapping {le是编辑距离，df是文本相似}