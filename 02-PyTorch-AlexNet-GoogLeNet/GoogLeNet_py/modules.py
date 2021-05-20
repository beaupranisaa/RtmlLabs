
import torch
import torch.nn as nn

class Inception(nn.Module):
    '''
    Inception block for a GoogLeNet-like CNN

    Attributes
    ----------
    in_planes : int
        Number of input feature maps
    n1x1 : int
        Number of direct 1x1 convolutions
    n3x3red : int
        Number of 1x1 reductions before the 3x3 convolutions
    n3x3 : int
        Number of 3x3 convolutions
    n5x5red : int
        Number of 1x1 reductions before the 5x5 convolutions
    n5x5 : int
        Number of 5x5 convolutions
    pool_planes : int
        Number of 1x1 convolutions after 3x3 max pooling
    b1 : Sequential
        First branch (direct 1x1 convolutions)
    b2 : Sequential
        Second branch (reduction then 3x3 convolutions)
    b3 : Sequential
        Third branch (reduction then 5x5 convolutions)
    b4 : Sequential
        Fourth branch (max pooling then reduction)
    '''
    
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        self.in_planes = in_planes
        self.n1x1 = n1x1
        self.n3x3red = n3x3red
        self.n3x3 = n3x3
        self.n5x5red = n5x5red
        self.n5x5 = n5x5
        self.pool_planes = pool_planes
        
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )
            

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1) # <<<< dim 1 = channel
#%%
class Aux_layer(nn.Module):
    def __init__(self, position):    
        super(Aux_layer, self).__init__()
        
        self.avgpool = nn.AvgPool2d(5, stride=3) 
        self.position = position
        
        if self.position == 'a4':
            self.conv = nn.Sequential(nn.Conv2d(512,128,kernel_size = 1,stride = 1), nn.ReLU(True))   
        else:
            self.conv = nn.Sequential(nn.Conv2d(528,128,kernel_size = 1,stride = 1), nn.ReLU(True))   

        self.fc1 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(True), nn.Dropout(0.7))
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self,x):
        aux = self.avgpool(x)
        aux = self.conv(aux)
#         print(f'aux conv: {aux.shape}')
        aux = aux.flatten(start_dim = 1)
#         print(f'aux: {aux.shape}')
        aux = self.fc1(aux)
        aux = self.fc2(aux)
        return aux

#%%
class GoogLeNet(nn.Module):
    '''
    GoogLeNet-like CNN

    Attributes
    ----------
    pre_layers : Sequential
        Initial convolutional layer
    a3 : Inception
        First inception block
    b3 : Inception
        Second inception block
    maxpool : MaxPool2d
        Pooling layer after second inception block
    a4 : Inception
        Third inception block
    b4 : Inception
        Fourth inception block
    c4 : Inception
        Fifth inception block
    d4 : Inception
        Sixth inception block
    e4 : Inception
        Seventh inception block
    a5 : Inception
        Eighth inception block
    b5 : Inception
        Ninth inception block
    avgpool : AvgPool2d
        Average pool layer after final inception block
    linear : Linear
        Fully connected layer
    '''

    def __init__(self):
        super(GoogLeNet, self).__init__()
        
        self.is_debug = False
        
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=7, stride =2, padding =3),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride =2, padding = 1),
            nn.LocalResponseNorm(5, alpha = 0.0001, beta = 0.75, k = 2.0),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(True),
            nn.LocalResponseNorm(5, alpha = 0.0001, beta = 0.75, k = 2.0),              
            nn.MaxPool2d(3, stride =2, padding =1),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool= nn.AvgPool2d(7, stride=1) ## orig = 8
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, 10)
        
        self.aux_a4 = Aux_layer('a4')
        self.aux_d4 = Aux_layer('d4')
    
    def forward(self, x):
        out = self.pre_layers(x)
        
        if self.is_debug : print(f'pre-layer: {out.shape}')
            
        out = self.a3(out)
        if self.is_debug : print(f'a3: {out.shape}')
            
        out = self.b3(out)
        if self.is_debug : print(f'b3 :{out.shape}')
            
        out = self.maxpool(out)
        if self.is_debug : print(f'maxpool: {out.shape}')
            
        out = self.a4(out)
        if self.is_debug : print(f'a4: {out.shape}')
            
        aux_a4 = self.aux_a4(out)
        out = self.b4(out)
        if self.is_debug : print(f'b4 : {out.shape}')
            
        out = self.c4(out)
        if self.is_debug : print(f'c4 : {out.shape}') 
            
        out = self.d4(out)
        if self.is_debug : print(f'd4 : {out.shape}')
            
        aux_d4 = self.aux_d4(out)
        out = self.e4(out)
        if self.is_debug : print(f'e4 : {out.shape}')
            
        out = self.maxpool(out)
        if self.is_debug : print(f'maxpool : {out.shape}')
            
        out = self.a5(out)
        if self.is_debug : print(f'a5 : {out.shape}')
            
        out = self.b5(out)
        if self.is_debug : print(f'b5 : {out.shape}')
            
        out = self.avgpool(out)
        if self.is_debug : print(f'avgpool : {out.shape}')
            
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        if self.training == True:
            return out, aux_a4, aux_d4
        else:
            return out

