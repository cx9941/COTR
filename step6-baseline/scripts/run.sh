for seed in 0 1
do
for turn in 0 1 2 3 4 5 6 7
do
for dataset_name in 'en' 'eu' 'jp'
do
python 2.llm_select.py --turn $turn --dataset_name $dataset_name --seed $seed
python 3.result_clean.py --turn $turn --dataset_name $dataset_name --seed $seed
done
done
done
