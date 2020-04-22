import torch.nn as nn
import torch.nn.functional as F

# 3x3 convolution
def Conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
           nn.Conv2d(in_channels= in_channels, out_channels=out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False),
           nn.BatchNorm2d(out_channels),
           nn.ReLU()
    )

def Conv2d(in_channels, out_channels,stride=1):
  return nn.Sequential(
         nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size= 3, padding=1, bias=False,
                  stride=stride)
    )

def Resnet(in_channels, out_channels, stride=1):
  return nn.Sequential(
           nn.Conv2d(in_channels= in_channels, out_channels=out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(),

           nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False),
           nn.BatchNorm2d(out_channels),
           nn.ReLU()
    )

def Maxpooling(kernel):
  return nn.MaxPool2d(kernel, kernel)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.PrepLayer = Conv3x3(3,64,stride=1)
        self.layer1 =  Conv2d(64,128 ,stride=1)
        self.pool1 = Maxpooling(2)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.resnet1 = Resnet(128,128,stride=1)
        self.layer2 = Conv2d(128,256,stride=1)
        self.pool2 = Maxpooling(2)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer3 = Conv2d(256, 512, stride=1)
        self.pool3 = Maxpooling(2)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(inplace=True)
        self.resnet2  = Resnet(512,512, stride=1)
        self.pool4 = Maxpooling(4)
        self.fc_layers = nn.Linear(512, 10)

    def forward(self, x):
        X  = self.PrepLayer(x) #input
        X  = self.layer1(X) # layer1
        X = self.pool1(X)# layer1
        X  = self.relu1(self.bn1(X))# layer1
        R1 = self.resnet1(X) # layer1
        X  = self.layer2(X+R1) # layer2
        X  = self.pool2(X)# layer2
        X  = self.relu2(self.bn2(X)) #layer2
        X  = self.layer3(X)  # layer3
        X  = self.pool3(X)  # layer3
        X  = self.relu3(self.bn3(X))  # layer3
        R2 = self.resnet2(X)# layer3
        X  = self.pool4(X+R2) # MaxPooling
        X  = X.view(-1, 512) #Output
        X = self.fc_layers(X)  # FC
        return F.log_softmax(X, dim=-1)

net = Net()
