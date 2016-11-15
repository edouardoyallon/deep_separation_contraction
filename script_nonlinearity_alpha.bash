for ALPHA in 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05 0
do
  python train_cifar.py --alpha $ALPHA --n_channel 32 --dataset cifar10
done

for ALPHA in 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05 0
do
  python train_cifar.py --alpha $ALPHA --n_channel 128 --dataset cifar10
done

for N in 16 64 256 512
do
  python train_cifar.py --alpha 1 --n_channel $N --dataset cifar10
done

for N in 16 32 64 128 256 512
do
  python train_cifar.py --alpha 1 --n_channel $N --dataset cifar100
done

python layerwise_statistic_cifar.py --alpha 1 --n_channel 32 --dataset cifar10
