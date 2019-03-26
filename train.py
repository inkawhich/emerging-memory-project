# NAI

# Main function to train MNIST models with or without adversarial training
# This trainer assumes we always want to train from scratch

# Boilerplate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# Custom
import models
import attack
import helpers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################
# Inputs
##################################################################
batch_size = 32
test_batch_size = 32
num_epochs = 2
num_workers = 4
save_model = True
checkpoint = "checkpoints/sample_model_checkpoint.pth.tar"

# Adversarial Training Inputs
do_AT = True
#AT_eps = 0.3; AT_alpha = 0.01; AT_iters = 40 # Official Madry Config
AT_eps = 0.3; AT_alpha = 0.05; AT_iters = 10

##################################################################
# Load Model
##################################################################
#net = models.all_fc().to(device)
#net = models.all_conv().to(device)
net = models.conv_and_fc().to(device)

net.train()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

print(net)

##################################################################
# Load Data
##################################################################
# Note: Not doing any normalization or data augmentation right now

# Load train data
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
	batch_size=batch_size, shuffle=True, num_workers=num_workers,
	)
# Load test data
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
	batch_size=test_batch_size, shuffle=False, num_workers=num_workers,
	)

##################################################################
# Training Loop
##################################################################

# Epoch Loop
for epoch in range(num_epochs):

	# Stat keepers
	train_loss_sum = 0.
	train_acc_sum = 0.
	cnt = 0.

	# Data Loop
	for idx,(data,labels) in enumerate(train_loader):
		# Prepare data
		data,labels = data.to(device),labels.to(device)

		# Attack data
		if do_AT:
			#data = attack.FGSM_attack(net, device, data, labels, eps=AT_eps)
			data = attack.PGD_attack(net, device, data, labels, eps=AT_eps, alpha=AT_alpha, iters=AT_iters)

		# Visualize
		#plt.imshow(data.clone().detach().cpu().numpy()[0,0],cmap='gray'); plt.show()

		# Clear previous gradients
		optimizer.zero_grad()
		net.zero_grad()
		# Forward pass data through model
		output = net(data)
		# Calculate Loss
		loss = criterion(output,labels)
		# Calculate gradients w.r.t parameters
		loss.backward()
		# Update parameters
		optimizer.step()

		# Update stat keepers
		pred = output.argmax(dim=1, keepdim=True)
		correct = pred.eq(labels.view_as(pred)).sum().item()
		train_acc_sum += correct
		train_loss_sum += loss.item()
		cnt += labels.size(0)

		if idx%100 == 0:
			print("[{}/{}][{}/{}] train_loss: {:.6f} train_acc: {:.6f}".format(
				epoch,num_epochs,idx,len(train_loader),
				train_loss_sum/cnt, train_acc_sum/cnt))
	
	# End of epoch. Run full test step	
	valacc,valloss = helpers.test_model(net,device,test_loader)
	print("[{}/{}] val_loss: {:.6f} val_acc: {:.6f}".format(
		epoch,num_epochs,valloss,valacc))

	# Save model
	if save_model:
		torch.save(net.state_dict(),checkpoint)
			








