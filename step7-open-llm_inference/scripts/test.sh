for seed in 0
do
for dataset_name in 'en' 'eu' 'jp'
do
for model_name in 'baichuan' 'baichuan-sft' 'nanbeige' 'nanbeige-sft' 'llama' 'llama-sft'
do
for thred in 50
do
python 2.result_get.py --model_name $model_name --dataset_name $dataset_name --seed $seed --thred $thred
done
done
done
done