xp_dir=../log/$1
pca=$2
mnist_normal=$3
mnist_outlier=$4

mkdir $xp_dir;

# MNIST training
python baseline_kde.py --xp_dir $xp_dir --dataset mnist --kernel exponential \
    --gridsearchcv 0 --out_frac 0 --pca $pca --mnist_normal $mnist_normal \
    --mnist_outlier $mnist_outlier;
