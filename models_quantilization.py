import torch.nn as nn
import quantilization
# All models input NCHW=[N,1,28,28]
# All models return logits for use with a cross-entropy loss



##########################################################
# Conv + FC + Quantilization
##########################################################
# From official pytorch/mnist example
class conv_and_fc_quan(nn.Module):
    def __init__(self, nbits):
        super(conv_and_fc_quan, self).__init__()
        self.nbits = nbits
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
        #print(x.size())
        for ind,(name,param) in enumerate(self.named_parameters()):
            s[ind] = param.data
            param.data = quantilization.Quantizer(self.nbits).apply(param.data)
            #param.data = quantilization.Quantizer_nonlinear(self.nbits).apply(param.data)
        x = self.features(x)
        x = x.view(-1, 4*4*50)
        x = self.classifier(x)
        for ind,(name,param) in enumerate(self.named_parameters()):
            param.data = s[ind]
        return x
