device=$1
xp_dir=../log/$2
solver=$3
lr=$4
n_epochs=$5
weight_dict_init=$6
hard_margin=$7
nu=$8
c_mean_init=$9
mnist_normal=${10}
mnist_outlier=${11}

mkdir $xp_dir;

# MNIST training
python baseline.py --device $device --xp_dir $xp_dir --dataset mnist \
    --solver $solver --loss svdd --lr $lr --lr_decay 0 --n_epochs $n_epochs \
    --batch_size 200 --nu $nu --out_frac 0 --weight_decay 0 --C 1e3 --gcn 1 \
    --unit_norm_used l1 --weight_dict_init $weight_dict_init \
    --in_name 0_mnist_adam_ce --hard_margin $hard_margin \
    --c_mean_init $c_mean_init --leaky_relu 1 --mnist_bias 0 \
    --mnist_normal $mnist_normal --mnist_outlier $mnist_outlier;
