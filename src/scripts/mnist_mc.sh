device=$1
xp_dir=../log/$2
solver=$3
lr=$4

mkdir $xp_dir;
> $xp_dir/log_mnist.txt;

# MNIST training
python baseline.py --device $device --xp_dir $xp_dir --dataset mnist \
    --solver $solver --loss svm --lr $lr --n_epochs 200 --batch_size 200 \
    --weight_decay 1 --C 1000 --gcn 1 --unit_norm_used l1 --weight_dict_init 1 \
    --ad_experiment 0 --leaky_relu 1 --mnist_bias 0;
