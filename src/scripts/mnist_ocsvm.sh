xp_dir=../log/$1
nu=$2
pca=$3
mnist_normal=$4
mnist_outlier=$5

mkdir $xp_dir;

# MNIST training
python baseline_ocsvm.py --xp_dir $xp_dir --dataset mnist \
    --loss OneClassSVM --kernel rbf --nu $nu --out_frac 0 --pca $pca \
    --mnist_normal $mnist_normal --mnist_outlier $mnist_outlier;
