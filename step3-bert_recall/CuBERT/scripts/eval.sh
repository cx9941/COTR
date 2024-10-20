for i in $(seq 0 1 10)
do
  echo $i
  python eval.py --k $i --bert_mode white
done