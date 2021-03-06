import torch
import torch.nn as nn
import torch.nn.functional as F
import attack
import quantilization
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
#AT_eps = 0.2;AT_alpha = 0.03; AT_iters = 10
#AT_eps = 0.15;AT_alpha = 0.02; AT_iters = 10
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
                #dat = attack.PGD_attack(model, device, dat, lbl, eps=AT_eps, alpha=AT_alpha, iters=AT_iters)
                dat = attack.PGD_attack(model,device,dat,lbl,eps=AT_eps,alpha=AT_alpha,iters=AT_iters)
                output = model(dat)
                loss_sum += F.cross_entropy(output,lbl).item()
                pred = output.argmax(dim=1, keepdim=True)
                #print(pred)
                #print(lbl.view_as(pred))
                correct = pred.eq(lbl.view_as(pred)).sum().item()
                #print(correct)
                acc_sum += correct
                cnt += lbl.size(0)
	model.train()
	return acc_sum/cnt,loss_sum/cnt
#test data and apply quantilization

def quan_param(model,nbits,do_linear):
    s = {}
    for index,param in enumerate(model.parameters()):
        s[index]=param.data
        if do_linear:
            param.data=quantilization.Quantizer(nbits).apply(param.data,nbits)
        else:
                #print(param.data)
                param.data=quantilization.Quantizer_nonlinear(nbits).apply(param.data,nbits)
                #print(param.data)
    return s
def recover_param(model,s):
    for index,param in enumerate(model.parameters()):
        param.data=s[index]
def print_param(model):
    for index,param in enumerate(model.parameters()):
            torch.set_printoptions(profile="full")
            print(param.data)
            torch.set_printoptions(profile="default")
def test_model_quan(model,device,loader,nbits,do_linear):
    s=quan_param(model,nbits,do_linear)
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
    recover_param(model,s)
    model.train()
    return acc_sum/cnt,loss_sum/cnt

def test_model_quan_AT(model,device,loader,nbits,do_linear):
    #print("origin data: ")
    #print_param(model)
    model.eval()
    s=quan_param(model,nbits,do_linear)
    #print("transform data: ")
    #print_param(model)
    acc_sum = 0.;loss_sum = 0.;cnt = 0.
    #with torch.no_grad():
    for dat,lbl in loader:
            dat,lbl = dat.to(device),lbl.to(device)
            dat = attack.PGD_attack(model, device, dat, lbl, eps=AT_eps, alpha=AT_alpha, iters=AT_iters)
            output = model(dat)
            #print(output)
            #print(lbl)
            loss_sum += F.cross_entropy(output,lbl).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(lbl.view_as(pred)).sum().item()
            #print(correct)
            acc_sum += correct
            cnt += lbl.size(0)
    model.train()
    recover_param(model,s)
    #print("recover data:")
    #print_param(model)
    return acc_sum/cnt,loss_sum/cnt
def test_model_quan_AT_2(model,device,loader,nbits,do_linear,alpha,eps):
    #print("origin data: ")                                                     
    #print_param(model)                                                         
    model.eval()
    s=quan_param(model,nbits,do_linear)
    #print("transform data: ")                                                  
    #print_param(model)                                                         
    acc_sum = 0.;loss_sum = 0.;cnt = 0.
    #with torch.no_grad():                                                      
    for dat,lbl in loader:
            dat,lbl = dat.to(device),lbl.to(device)
            dat = attack.PGD_attack(model, device, dat, lbl, eps=eps, alpha=
alpha, iters=AT_iters)
            output = model(dat)
            #print(output)                                                      
            #print(lbl)                                                         
            loss_sum += F.cross_entropy(output,lbl).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(lbl.view_as(pred)).sum().item()
            #print(correct)                                                     
            acc_sum += correct
            cnt += lbl.size(0)
    model.train()
    recover_param(model,s)
    #print("recover data:")                                                     
    #print_param(model)                                                         
    return acc_sum/cnt,loss_sum/cnt
