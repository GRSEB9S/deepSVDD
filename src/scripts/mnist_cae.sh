device=$1
xp_dir=../log/$2
solver=$3
lr=$4
n_epochs=$5
unit_norm_used=$6
mnist_rep_dim=$7
weight_dict_init=$8
ae_loss=$9
mnist_normal=${10}
mnist_outlier=${11}

mkdir $xp_dir;

# MNIST training
python baseline.py --device $device --xp_dir $xp_dir --dataset mnist \
    --solver $solver --loss autoencoder --lr $lr --lr_decay 0 \
    --n_epochs $n_epochs --batch_size 200 --out_frac 0 --ae_weight_decay 1 \
    --ae_C 1e3 --gcn 1 --unit_norm 1 --unit_norm_used $unit_norm_used \
    --mnist_rep_dim $mnist_rep_dim --weight_dict_init $weight_dict_init \
    --leaky_relu 1 --ae_loss $ae_loss --mnist_bias 0 \
    --mnist_normal $mnist_normal --mnist_outlier $mnist_outlier;
