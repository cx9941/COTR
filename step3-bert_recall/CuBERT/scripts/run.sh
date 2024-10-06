set -o errexit
export CUDA_VISIBLE_DEVICES='5'
for learning_rate in 2e-5 1e-5 5e-6 1e-6
do
for batch_size in 32 64
do
# python main.py --learning_rate $learning_rate --batch_size $batch_size
for test_dataset_name in 'eu' 'en'
do
python test.py --learning_rate $learning_rate --batch_size $batch_size --test_dataset_name $test_dataset_name
done
done
done