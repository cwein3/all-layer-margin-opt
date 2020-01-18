import argparse
import os
import shutil
from datetime import datetime

import torch 
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn

import save_util
import train_util
import data_util
#import eval_loss_util
import pickle

import models as models

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

lr_scheds= ['wr_default']
reg_types=['none', 'adv_full']
parser = argparse.ArgumentParser(description='Adversarial-(ish) Hidden Layer Training')
parser.add_argument('--epochs', default=200, type=int,
					help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
					help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=128, type=int,
					help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
					help='weight decay (default: 5e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
					help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
					help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='ResNet', type=str,
					help='name of experiment')
parser.add_argument('--dataset', choices=["cifar10", "cifar100"], default="cifar10",
					help='cifar10 or cifar100')
parser.add_argument('--corrupt_prob', type=float, default=0, help='Probability of corrupting label.')
parser.add_argument('--lr_sched', choices=lr_scheds, default='wr_default', 
					help=' | '.join(lr_scheds))
parser.add_argument('--arch', choices=model_names, default="bn_resnet56",
					help='model architecture:' + ' | '.join(model_names))
parser.add_argument('--reg_type', choices=reg_types, default="none", help='reg type:' + ' | '.join(reg_types))
parser.add_argument('--augment', action="store_true", help="whether to augment data")
parser.add_argument('--inner_steps', type=int, default=1, help='Number of steps of inner maximization procedure to run.')
parser.add_argument('--inner_lr', type=float, default=0.01, help='the learning rate we use to perturb the hidden layer')
parser.add_argument('--inner_wd', type=float, default=0,help='weight decay in the inner maximization.')
parser.add_argument('--switch_time', type=int, default=0 ,help='when to switch the loss to this adversarial formulation')
parser.add_argument('--save_dir', type=str, help='directory for saving the experimental run')
parser.add_argument('--data_dir', type=str, help='directory where the data is saved')
parser.set_defaults(augment=False)

def main():
	args = parser.parse_args()
	for arg in vars(args):
		print(arg, " : ", getattr(args, arg))
	save_str = "arch_%s_reg_%s" % (
		args.arch,
		args.reg_type)
	save_dir = os.path.join(args.save_dir, save_str)
	
	train_loader, val_loader = data_util.load_data(
		args.batch_size, 
		args.dataset,
		data_path=args.data_dir,
		corrupt_prob=args.corrupt_prob,
		augment=args.augment)


	print("=> creating model '{}'".format(args.arch))
	model_args = {
		"num_classes": 10 if args.dataset == "cifar10" else 100	
	}
	
	model = models.__dict__[args.arch](**model_args)

	print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))

	model = model.cuda()

	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_val = checkpoint['best_val']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				.format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	criterion = nn.CrossEntropyLoss().cuda()
	optim_hparams = {
		'base_lr' : args.lr, 
		'momentum' : args.momentum,
		'weight_decay' : args.weight_decay
	}
	lr_hparams = {'lr_sched' : args.lr_sched}

	optimizer = train_util.create_optimizer(
		model,
		optim_hparams)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	save_util.write_args(args, save_dir)
	scalar_summary_file = os.path.join(save_dir, "scalars.txt")
	data_dict_file = os.path.join(save_dir, "data_dict.pkl")
	scalar_dict = {}
	best_val = 0
	all_dict = {}

	for epoch in range(args.start_epoch, args.epochs):
		lr = train_util.adjust_lr(
			optimizer,
			epoch + 1,
			args.lr,
			lr_hparams)

		reg_type = 'none' if epoch + 1 < args.switch_time else args.reg_type

		train_hparams = {
			"inner_lr" : args.inner_lr,
			"inner_step" : args.inner_steps,
			"lr" : lr,
			"reg_type" : reg_type,
			"inner_wd" : args.inner_wd
		}

		train_acc, train_loss,perturb_acc = train_util.train_loop(
			train_loader,
			model,
			criterion,
			optimizer,
			epoch,
			train_hparams,
			print_freq=args.print_freq)

		print("Validating clean accuracy.")
		val_acc, val_loss = train_util.validate(
			val_loader,
			model,
			criterion,
			epoch,
			print_freq=args.print_freq)

		scalar_epoch = {
			"lr": lr, 
			"inner_lr": args.inner_lr,
			"inner_steps" : args.inner_steps,
			"train_loss": train_loss, 
			"train_acc": train_acc, 
			"val_loss": val_loss, 
			"val_acc": val_acc,
			"perturb_acc": perturb_acc,
			"reg_type" : reg_type
		}

		scalar_dict[epoch + 1] = scalar_epoch

		save_util.log_scalar_file(
			scalar_epoch,
			epoch + 1,
			scalar_summary_file)

		is_best = val_acc > best_val
		best_val = max(val_acc, best_val)

		save_util.save_checkpoint(
			{
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_val': best_val,
			}, 
			scalar_dict,
			is_best,
			save_dir)

		print('Best accuracy: ', best_val)
			
main()

