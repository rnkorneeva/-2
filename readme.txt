Корнеева Ирина Александровна 320
Реализованы все части задания!
Чтобы запустить Вам необходимо:
	1) make
	2) build/bin/task2 -d data/multiclass/train_labels.txt  -m model.txt --train
	3) build/bin/task2 -d data/multiclass/test_labels.txt -m model.txt -l predictions.txt --predict
	4) python compare.py data/multiclass/test_labels.txt predictions.txt
Какая получилась точноть?
Precision: 0.787879 - base
Precision: 0.868687 - base + part1
Precision: 0.909091 - base + part1 + part2
