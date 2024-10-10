set -o errexit
export CUDA_VISIBLE_DEVICES="2,3"
for seed in 0 1 2
do
for backbone in 'Meta-Llama-3.1-8B-Instruct'
do
for dataset_name in 'en' 'eu' 'jp'
do
accelerate launch --num_processes=2 --main_process_port=7889  1.llm_select.py --backbone $backbone --dataset_name $dataset_name --seed $seed
done
done
done