xp_dir=../log/$1
pca=$2
cifar10_normal=$3
cifar10_outlier=$4

mkdir $xp_dir;

# CIFAR-10 training
python baseline_kde.py --xp_dir $xp_dir --dataset cifar10 --pca $pca \
    --kernel exponential --gridsearchcv 0 --out_frac 0 --unit_norm_used l1 \
    --gcn 1 --cifar10_normal $cifar10_normal --cifar10_outlier $cifar10_outlier;
