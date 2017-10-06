device=$1
xp_dir=../log/$2
solver=$3
lr=$4
weight_dict_init=$5
hard_margin=$6
bedroom_n_train=$7
bedroom_monitor_int=$8
seed=$9

mkdir $xp_dir;

# Bedroom training
python baseline.py --device $device --xp_dir $xp_dir --dataset bedroom \
    --solver $solver --loss svdd --lr $lr --n_epochs 1 --gcn 1 \
    --weight_dict_init $weight_dict_init --batch_size 100 --nu 0.1 \
    --out_frac 0 --hard_margin $hard_margin --weight_decay 1 --C 1e6 \
    --c_mean_init 0 --unit_norm_used l1 --bedroom_n_train $bedroom_n_train \
    --bedroom_monitor_int $bedroom_monitor_int --seed $seed;
