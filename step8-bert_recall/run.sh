for i in 0 1 2 3 4 5
do
nohup python 3.Levenshtein_distance.py --process_idx $i --process_num 6 > logs/Levenshtein_distance_$i.log &
done