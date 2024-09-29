set -o errexit
for seed in 1 2 3
do
export CUDA_VISIBLE_DEVICES="3,5,6"
for turn in 0 1
do
accelerate launch --num_processes=2 --main_process_port=7888 1.llm_select.py --turn $turn --seed $seed
python 2.result_clean.py  --turn $turn
done
done
