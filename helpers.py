import torch
import torch.nn as nn
import torch.nn.functional as F

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