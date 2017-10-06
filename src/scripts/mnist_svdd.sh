device=$1
xp_dir=../log/$2
solver=$3
lr=$4
n_epochs=$5
weight_decay=$6
weight_dict_init=$7
hard_margin=$8
nu=$9
c_mean_init=${10}
mnist_normal=${11}
mnist_outlier=${12}

mkdir $xp_dir;

# MNIST training
python baseline.py --device $device --xp_dir $xp_dir --dataset mnist \
    --solver $solver --loss svdd --lr $lr --lr_decay 1 \
    --lr_decay_after_epoch 20 --n_epochs $n_epochs \
    --batch_size 200 --nu $nu --out_frac 0 --weight_decay $weight_decay \
    --C 1e3 --gcn 1 --unit_norm_used l1 --weight_dict_init $weight_dict_init \
    --hard_margin $hard_margin --c_mean_init $c_mean_init --leaky_relu 1 \
    --mnist_bias 0 --mnist_normal $mnist_normal --mnist_outlier $mnist_outlier;
