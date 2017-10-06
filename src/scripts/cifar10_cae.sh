device=$1
xp_dir=../log/$2
solver=$3
lr=$4
n_epochs=$5
weight_dict_init=$6
ae_loss=$7
cifar10_rep_dim=$8
cifar10_normal=$9
cifar10_outlier=${10}


mkdir $xp_dir;

# MNIST training
python baseline.py --device $device --xp_dir $xp_dir --dataset cifar10 \
    --solver $solver --loss autoencoder --lr $lr --lr_decay 0 \
    --n_epochs $n_epochs --weight_dict_init $weight_dict_init --batch_size 200 \
    --nu 0.1 --out_frac 0 --ae_loss $ae_loss --ae_weight_decay 1 --ae_C 1e6 \
    --c_mean_init 0 --unit_norm 1 --unit_norm_used l1 --gcn 1 --pretrain 0 \
    --leaky_relu 1 --cifar10_bias 0 --cifar10_rep_dim $cifar10_rep_dim \
    --cifar10_normal $cifar10_normal --cifar10_outlier $cifar10_outlier;
