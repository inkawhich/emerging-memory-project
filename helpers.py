import torch
import torch.nn as nn
import torch.nn.functional as F
import attack
# class AverageMeter(object):
# 	"""Computes and stores the average and current value"""
# 	def __init__(self):
# 		self.reset()
# 	def reset(self):
# 		self.val = 0
# 		self.avg = 0
# 		self.sum = 0
# 		self.count = 0
# 	def update(self, val, n=1):
# 		self.val = val
# 		self.sum += val * n
# 		self.count += n
# 		self.avg = self.sum / self.count

AT_eps = 0.3; AT_alpha = 0.05; AT_iters = 10
#test data using original data
def test_model(model,device,loader):
	model.eval()
	acc_sum = 0.;loss_sum = 0.;cnt = 0.
	with torch.no_grad():
		for dat,lbl in loader:
			dat,lbl = dat.to(device),lbl.to(device)
			output = model(dat)
			loss_sum += F.cross_entropy(output,lbl).item()
			pred = output.argmax(dim=1, keepdim=True)
			correct = pred.eq(lbl.view_as(pred)).sum().item()
			acc_sum += correct
			cnt += lbl.size(0)
	model.train()
	return acc_sum/cnt,loss_sum/cnt

#test data using AT data
def test_model_AT(model,device,loader):
	model.eval()
	acc_sum = 0.;loss_sum = 0.;cnt = 0.
	#with torch.no_grad():
	for dat,lbl in loader:            
		dat,lbl = dat.to(device),lbl.to(device)
		#dat = attack.PGD_attack_test(model, device, dat, lbl, eps=AT_eps, alpha=AT_alpha, iters=AT_iters)
		dat = attack.PGD_attack(model, device, dat, lbl, eps=AT_eps, alpha=AT_alpha, iters=AT_iters)
		output = model(dat)
		loss_sum += F.cross_entropy(output,lbl).item()
		pred = output.argmax(dim=1, keepdim=True)
		correct = pred.eq(lbl.view_as(pred)).sum().item()
		acc_sum += correct
		cnt += lbl.size(0)
	model.train()
	return acc_sum/cnt,loss_sum/cnt

#test data and apply quantilization
    

######QUAN
    
def init_para(model_target,model_pretrain):
    for param_target,param_pretrain in zip(model_target.parameters(),model_pretrain.parameters()):
        param_target.data.copy_(param_pretrain.data)
        
def test_model_quan(model,model_quan,device,loader):
    init_para(model_quan,model)
    model_quan.eval()
    acc_sum = 0.;loss_sum = 0.;cnt = 0.
    with torch.no_grad():
        for dat,lbl in loader:
            dat,lbl = dat.to(device),lbl.to(device)
            output = model_quan(dat)
            loss_sum += F.cross_entropy(output,lbl).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(lbl.view_as(pred)).sum().item()
            acc_sum += correct
            cnt += lbl.size(0)
    model_quan.train()
    return acc_sum/cnt,loss_sum/cnt

def test_model_quan_AT(model,model_quan,device,loader):
    init_para(model_quan,model)
    model_quan.eval()
    acc_sum = 0.;loss_sum = 0.;cnt = 0.
    with torch.no_grad():
        for dat,lbl in loader:
            dat = attack.PGD_attack_test(model, device, dat, lbl, eps=AT_eps, alpha=AT_alpha, iters=AT_iters)
            dat,lbl = dat.to(device),lbl.to(device)
            output = model_quan(dat)
            loss_sum += F.cross_entropy(output,lbl).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(lbl.view_as(pred)).sum().item()
            acc_sum += correct
            cnt += lbl.size(0)
    model_quan.train()
    return acc_sum/cnt,loss_sum/cnt
