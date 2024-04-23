#!/usr/bin/env bash
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

exp_id=$1
tag=$2
num_eps=$3
# Set default value for debug if not provided
debug=${4:-run}

log_dir=logs/${exp_id}
mkdir -p $log_dir/node_info

debug_flag=""
if [ "$debug" = "debug" ]; then
    debug_flag="--debugpy"
fi

python -u cos_eor/trainer/hie_policy_runner.py \
    --num-eps $num_eps \
    --exp-config $log_dir/configs/${tag}.yaml \
    --tag ${tag} \
    --out-dir $log_dir \
    $debug_flag \
|& tee $log_dir/stdout-0-${tag}.log
