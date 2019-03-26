import torch.nn as nn

# All models input NCHW=[N,1,28,28]
# All models return logits for use with a cross-entropy loss


##########################################################
# FC only
##########################################################
class all_fc(nn.Module):
    def __init__(self):
        super(all_fc,self).__init__()
        self.features = nn.Sequential(
            nn.Linear(28*28, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
        )
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.features(x)
        return x

##########################################################
# Conv + FC
##########################################################
# From official pytorch/mnist example
class conv_and_fc(nn.Module):
    def __init__(self):
        super(conv_and_fc, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            )
        self.classifier = nn.Sequential(
            nn.Linear(4*4*50, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4*4*50)
        x = self.classifier(x)
        return x

##########################################################
# Conv Only
##########################################################
class all_conv(nn.Module):
    def __init__(self):
        super(all_conv,self).__init__()
        self.classes = 10
        self.features = nn.Sequential(
            # N,1,28,28
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # N,16,14,14
            nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # N,32,7,7
            nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # N,16,3,3
            nn.Conv2d(16,10,kernel_size=3,stride=1,padding=0),
            # N,10,1,1
        )   
    def forward(self,x):
        # Forward pass through conv layers
        x = self.features(x)
        # Reshape for use with softmax
        x = x.view(x.size(0),self.classes)
        return x
    
