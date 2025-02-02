set -o errexit
export CUDA_VISIBLE_DEVICES='0'
for learning_rate in 2e-5
do
for batch_size in 32
do
for dataset_name in 'en'
do
# python main.py --learning_rate $learning_rate --batch_size $batch_size --dataset_name $dataset_name --bert_mode white --mode 'train'
python test.py --learning_rate $learning_rate --batch_size $batch_size --dataset_name $dataset_name --bert_mode white --mode 'test'
done
done
done