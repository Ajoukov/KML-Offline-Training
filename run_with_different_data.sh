#/usr/bin/sh

# for benchmark in "mixgraph" "updaterandom" "readrandom" "readrandomwriterandom" "readwhilewriting" "readreverse" "readseq" "fillseq" "fill100k"
for benchmark in "readrandom" "readrandomwriterandom" "readwhilewriting" "readreverse" "readseq" "fillseq" "fill100k"
do
	cp input_data/$benchmark.csv input_data/inputs.csv
	python3 table_latency_use_N.py
	i=0
	while [ $i -ne 3 ]
	do
		python3 table_latency_use_N.py > /dev/null
		python3 train_ml_latency_use_N.py > /dev/null

		echo "$benchmark" >> run_w_diff.out
		python3 test_ml_latency_use_N.py >> run_w_diff.out
		cp output_data/box_plots_predictions_with_percentiles.png output_data/$benchmark-1-$i.png
		i=$((i+1))
	done
done
