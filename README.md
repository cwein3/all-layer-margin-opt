# all-layer-margin-opt
All-layer margin optimization code. 

To run the algorithm on cifar10, you could use the following command:

``python train_cifar.py --lr 0.1 --dataset cifar10 --epochs 200 --arch bn_wideresnet16 --switch_time 1 --inner_lr 0.01 --inner_steps 1 --augment --reg_type adv_full --save_dir <where you want to save the model> --data_dir <path to cifar10 data>``
