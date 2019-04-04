import torch.nn as nn
import quantilization
# All models input NCHW=[N,1,28,28]
# All models return logits for use with a cross-entropy loss

##########################################################
# Conv + FC + Quantilization
##########################################################
# From official pytorch/mnist example
class conv_and_fc_quan(nn.Module):
    def __init__(self, nbits,do_linear):
        super(conv_and_fc_quan, self).__init__()
        self.nbits = nbits
        self.do_linear = do_linear
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
        s = {}
        for ind,(name,param) in enumerate(self.named_parameters()):
            s[ind] = param.data
            if self.do_linear:
                param.data = quantilization.Quantizer(self.nbits).apply(param.data,self.nbits)
            else:
                param.data = quantilization.Quantizer_nonlinear(self.nbits).apply(param.data,self.nbits)
        x = self.features(x)
        x = x.view(-1, 4*4*50)
        x = self.classifier(x)
        for ind,(name,param) in enumerate(self.named_parameters()):
            param.data = s[ind]
        return x
##########################################################
# FC only + Quantilization        
##########################################################
class all_fc_quan(nn.Module):
    def __init__(self, nbits,do_linear):
        super(all_fc_quan,self).__init__()
        self.nbits = nbits
        self.do_linear = do_linear
        self.features = nn.Sequential(
            nn.Linear(28*28, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
        )
    def forward(self,x):
        s = {}
        for ind,(name,param) in enumerate(self.named_parameters()):
            s[ind] = param.data
            if self.do_linear:
                param.data = quantilization.Quantizer(self.nbits).apply(param.data,self.nbits)
            else:
                param.data = quantilization.Quantizer_nonlinear(self.nbits).apply(param.data,self.nbits)
        x = x.view(x.size(0),-1)
        x = self.features(x)
        for ind,(name,param) in enumerate(self.named_parameters()):
            param.data = s[ind]
        return x
##########################################################
# Conv Only + Quantilization        
##########################################################
class all_conv_quan(nn.Module):
    def __init__(self, nbits,do_linear):
        super(all_conv_quan,self).__init__()
        self.nbits = nbits
        self.do_linear = do_linear
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
        s = {}
        for ind,(name,param) in enumerate(self.named_parameters()):
            s[ind] = param.data
            if self.do_linear:
                param.data = quantilization.Quantizer(self.nbits).apply(param.data,self.nbits)
            else:
                param.data = quantilization.Quantizer_nonlinear(self.nbits).apply(param.data,self.nbits)
        # Forward pass through conv layers
        x = self.features(x)
        # Reshape for use with softmax
        x = x.view(x.size(0),self.classes)
        for ind,(name,param) in enumerate(self.named_parameters()):
            param.data = s[ind]
        return x
