#!/usr/bin/env sh

set -x
set -e

for BASE in $(seq 2 7) ; do
    FST=$((BASE * 2))
    SND=$((BASE * 2 + 1))

    curl -O http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${FST}.gz &
    curl -O http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${SND}.gz &
    wait

    gunzip day_${FST}.gz &
    gunzip day_${SND}.gz &
    wait

    python process_data.py ${FST} &
    python process_data.py ${SND} &
    wait

    rm day_${FST}
    rm day_${SND}

    for INDEX in $((FST * 2)) $((FST * 2 + 1)) $((SND * 2)) $((SND * 2 + 1)) ; do
        for PREFIX in eval_data eval_label train_label train_data ; do
            s3cmd put ${PREFIX}_${INDEX}.csv s3://mxnet-experiment/criteo-dataset/ &
        done
        wait
        for PREFIX in eval_data eval_label train_label train_data ; do
            rm ${PREFIX}_${INDEX}.csv
        done
    done
done
