for dataset_name in 'eu' 'jp' 'fr' 'en'
do
python 0_dataprocess.py --dataset_name $dataset_name
python 1_job_split.py --dataset_name $dataset_name
python 2_filter.py --dataset_name $dataset_name
done