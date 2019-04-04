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
import models_quantilization
import attack
import helpers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################
# Inputs
##################################################################
batch_size = 128
test_batch_size = 128
num_epochs = 10
num_workers = 8
save_model = False
checkpoint = "checkpoints/sample_model_checkpoint.pth.tar"

# Adversarial Training Inputs
#do_AT = True
do_AT=False
#AT_eps = 0.3; AT_alpha = 0.01; AT_iters = 40 # Official Madry Config
AT_eps = 0.3; AT_alpha = 0.05; AT_iters = 10


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
   


for nbits in range(1,9):
#for nbits in [4,8,12,16]:
    ##################################################################
    # Load Model
    ##################################################################
    
    #net = models.all_fc().to(device)
    #net = models.all_conv().to(device)
    net = models.conv_and_fc().to(device)
    do_quan = False
    print("nbits:{}".format(nbits)) 


    #net = models_quantilization.conv_and_fc_quan(nbits,do_linear=True).to(device)
    #net = models_quantilization.conv_and_fc_quan(nbits,do_linear=False).to(device)
    #do_quan = True
    #net = models_quantilization.all_fc_quan(nbits,do_linear=True).to(device)
    #net = models_quantilization.all_fc_quan(nbits,do_linear=False).to(device)
    #do_quan = True
    #net = models_quantilization.all_conv_quan(nbits,do_linear=True).to(device)
    #net = models_quantilization.all_conv_quan(nbits,do_linear=False).to(device)
    #do_quan = True 
    net.train()
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    
    print(net)
    
     
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
        if do_quan:
            valacc,valloss = helpers.test_model(net,device,test_loader)
        
            print("Clean dataset testing:[{}/{}] val_loss: {:.6f} val_acc: {:.6f}".format(
    		epoch,num_epochs,valloss,valacc))
            valacc_AT,valloss_AT = helpers.test_model_AT(net,device,test_loader)
            print("AT dataset testing:[{}/{}] val_loss: {:.6f} val_acc: {:.6f}".format(
    		epoch,num_epochs,valloss_AT,valacc_AT))

        else:
            valacc_quan,valloss_quan = helpers.test_model_quan(net,device,test_loader,nbits,do_linear=True)
            print("[Normal training Linear Quantilized testing]Clean dataset testing:[{}/{}] val_loss: {:.6f} val_acc: {:.6f}".format(epoch,num_epochs,valloss_quan,valacc_quan))
            valacc_quan_AT,valloss_quan_AT = helpers.test_model_quan_AT(net,device,test_loader,nbits,do_linear=True)
            print("[Normal training Linear Quantilized testing]AT dataset testing:[{}/{}] val_loss: {:.6f} val_acc: {:.6f}".format(epoch,num_epochs,valloss_quan_AT,valacc_quan_AT))
            valacc_quan,valloss_quan = helpers.test_model_quan(net,device,test_loader,nbits,do_linear=False)
            print("[Normal training NonLinear Quantilized testing]Clean dataset testing:[{}/{}] val_loss: {:.6f} val_acc: {:.6f}".format(epoch,num_epochs,valloss_quan,valacc_quan))
            valacc_quan_AT_nonlinear,valloss_quan_AT_nonlinear = helpers.test_model_quan_AT(net,device,test_loader,nbits,do_linear=False)
            print("[Normal training NonLinear Quantilized testing]AT dataset testing:[{}/{}] val_loss: {:.6f} val_acc: {:.6f}".format(epoch,num_epochs,valloss_quan_AT_nonlinear,valacc_quan_AT_nonlinear))
            #valacc_AT,valloss_AT = helpers.test_model_AT(net,device,test_loader)
            #print("AT dataset testing:[{}/{}] val_loss: {:.6f} val_acc: {:.6f}".format(epoch,num_epochs,valloss_AT,valacc_AT))
            #valacc,valloss = helpers.test_model(net,device,test_loader)
            #print("Clean dataset testing:[{}/{}] val_loss: {:.6f} val_acc: {:.6f}".format(epoch,num_epochs,valloss,valacc))
    # Save model
    #if save_model:
     #       torch.save(net.state_dict(),checkpoint)
    			








