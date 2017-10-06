device=$1
xp_dir=../log/$2
solver=$3
lr=$4
n_epochs=$5

echo "Using device" $device
echo "Writing to directory: " $xp_dir

mkdir $xp_dir;

# CIFAR-10 Adadelta training
python baseline.py --device $device --xp_dir $xp_dir --dataset cifar10 \
    --solver $solver --lr $lr --n_epochs $n_epochs --batch_size 200 --loss ce \
    --weight_decay 1 --C 1e6 --ad_experiment 0 --unit_norm_used l1 --gcn 1 \
    --leaky_relu 1 --cifar10_bias 0 --cifar10_rep_dim 64;
