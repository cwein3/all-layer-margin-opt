import math
import torch
import subprocess
import itertools
import numpy as np 
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

def update_step(criterion, optimizer, model, input_var, target_var, hparams):
	if hparams['reg_type'] == 'none':
		return standard_update(
			criterion,
			optimizer,
			model,
			input_var,
			target_var			
			)
	elif hparams['reg_type'] == 'adv_full':
		return adv_update_full(
			criterion,
			optimizer,
			model,
			input_var,
			target_var,
			hparams)


def standard_update(
	criterion,
	optimizer,
	model,
	inputs,
	labels):
	start_time = time.time()
	output = model(inputs)
	loss = criterion(output, labels)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss, output, time.time() - start_time, None

def adv_update_full(
	criterion,
	optimizer,
	model,
	inputs,
	labels,
	hparams):
	start_time = time.time()

	# first find all the necessary perturbations
	model_out, delta_vals = model.forward_init_delta(inputs)
	first_output = model_out
	delta_opt = torch.optim.SGD(delta_vals, lr=hparams['inner_lr'], weight_decay=hparams['inner_wd'])
	for step in range(hparams['inner_step']):
		if step > 0:
			model_out = model.forward_perturb(inputs, delta_vals)
		loss_val = -criterion(model_out, labels)
		delta_opt.zero_grad()
		loss_val.backward()
		delta_opt.step()

	# after several steps of this, we compute the update to the model
	delta_vals = [delta.detach() for delta in delta_vals]
	perturb_outs = model.forward_perturb(inputs, delta_vals)
	perturb_loss = criterion(perturb_outs, labels)
	loss = perturb_loss
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	return perturb_loss, first_output, time.time() - start_time, perturb_outs

	






