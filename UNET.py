import torch
import torch.nn as nn
import torch.utils as utils


def crop(x_crop,x_target):
    
    shape1= x_crop.size(2)
    shape2= x_target.size(2)

    factor = (shape1-shape2)//2

    return x_crop[:,:,factor: shape1 -factor, factor: shape1 - factor]


class Ublock(nn.Module):
    
    

    def __init__(self, inchannel, outchannel):
        
        super(Ublock,self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel,kernel_size=3,stride=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(in_channels=outchannel, out_channels=outchannel,kernel_size=3,stride=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
        )


    def forward(self, x):
        x = self.block(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_classes = 2, block = Ublock):
        super(UNet, self).__init__()

        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lev1 = block(1,64)
        self.lev2 = block(64,128)
        self.lev3 = block(128,256)
        self.lev4 = block(256,512)
        self.lev5 = block(512,1024)

        self.upsample1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,stride=2,kernel_size=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=512, out_channels=256,stride=2,kernel_size=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,stride=2,kernel_size=2)
        self.upsample4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,stride=2,kernel_size=2)

        self.dlev4=block(1024,512)
        self.dlev3=block(512,256)
        self.dlev2=block(256,128)
        self.dlev1=block(128,64)
        self.out =nn.Sequential(nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
                                 nn.Softmax())
    def forward(self, x):
        
        #encoder 
        #level 1
        x1=self.lev1(x)
        x=self.pool(x1)
        
        #level 2
        x2=self.lev2(x)
        x=self.pool(x2)
        
        #level 3
        x3=self.lev3(x)
        x=self.pool(x3)
        
        #level 4
        x4=self.lev4(x)
        x=self.pool(x4)
        
        #level 5
        x=self.lev5(x)

        #decoder
        #level 4
        x = self.upsample1(x)
        x4 = crop(x4,x)
        x = torch.cat((x,x4),dim=1)
        x = self.dlev4(x)

        #level 3
        x = self.upsample2(x)
        x3 = crop(x3,x)
        x = torch.cat((x,x3),dim=1)
        x = self.dlev3(x)

        #level 2
        x = self.upsample3(x)
        x2 = crop(x2,x)
        x = torch.cat((x,x2),dim=1)
        x = self.dlev2(x)

        #level 1
        x = self.upsample4(x)
        x1 = crop(x1,x)
        x = torch.cat((x,x1),dim=1)
        x = self.dlev1(x)


        x = self.out(x)
        return x
