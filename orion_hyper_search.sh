#!/bin/bash

DEFLT_EXP_NAME="ORION-HYPER-SEARCH"
DEFLT_MAX_TRIALS=20

if [ "$1" != "" ]; then
    EXP_NAME=$1
else
    EXP_NAME=$DEFLT_EXP_NAME
fi

if [ "$2" != "" ]; then
    MAX_TRIALS=$2
else
    MAX_TRIALS=$DEFLT_MAX_TRIALS
fi

echo $EXP_NAME
echo $MAX_TRIALS

orion -v hunt -n $EXP_NAME --max-trials $MAX_TRIALS ./hyper_search.py --mlflow_experiment_name $EXP_NAME --num_epochs 10 --num_max_chkpts 1 --output_dir /network/tmp1/fansitca/TT-Training/search_ouput --learning_rate~'loguniform(1e-5, 1.0)' --kernel_size~'choices([2, 3, 4])' --pool_size~'choices([2, 3, 4])' --num_fc~'choices([1, 2, 3])' --strides~'choices([1, 2, 3])' --dropout_rate~'uniform(0.0, 0.6)' --fc_size~'choices([256, 512, 1000, 1500, 2000])'