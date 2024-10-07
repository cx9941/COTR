set -o errexit
export CUDA_VISIBLE_DEVICES='0'
for learning_rate in 2e-5 1e-5 5e-6 1e-6
do
for batch_size in 32 64
do
python main.py --learning_rate $learning_rate --batch_size $batch_size --dataset_name 'jp'
# for dataset_name in 'jp'
# do
# python test.py --learning_rate $learning_rate --batch_size $batch_size --dataset_name $dataset_name
# done
done
done