import torch.nn as nn

class CNNmodel(nn.Module):
    def __init__(self,kernel,out_out,shape,Knums=[10,5],Pnums=[10]):
        super().__init__()
        self.Knums=Knums
        self.in_layer=nn.Sequential(
            nn.Conv2d(in_channels=kernel,out_channels=Knums[0],kernel_size=3,padding=1),
            nn.ReLU())
        self.h_layer=nn.ModuleList()
        for n in range(len(Knums)-2):
            self.h_layer.append(nn.Sequential(
                nn.Conv2d(in_channels=Knums[n],out_channels=Knums[n+1],kernel_size=3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)))
        self.h_layer.append(nn.Sequential(
            nn.Conv2d(in_channels=Knums[-2],out_channels=Knums[-1],kernel_size=3,padding=1),
            nn.BatchNorm2d(Knums[-1]),
            nn.ReLU(),
            nn.AvgPool2d(2)))
        self.fcs=nn.ModuleList()
        self.fcs.append(nn.Linear((int(((shape/(2**(len(Knums)-1)))**2 )*Knums[-1])),Pnums[0]))
        self.fcs.append(nn.ReLU())
        for n in range(len(Pnums)-1):
            self.fcs.append(nn.Linear(Pnums[n],Pnums[n+1]))
            self.fcs.append(nn.ReLU())
        self.fcs.append(nn.Linear(Pnums[-1],out_out))
    
    def forward(self,x):
        x=self.in_layer(x)
        for module in self.h_layer:
            x=module(x)
        x=x.contiguous().view(x.shape[0],-1)
        for module in self.fcs:
            x=module(x)
        return x