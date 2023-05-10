import torch
import torch.nn as nn

class TwoConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)
    
class Contracting(nn.Module):

    def __init__(self, in_channels, out_channels, p):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(p=p),
            TwoConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layer(x)
    
class Expansive(nn.Module):
    
    def __init__(self, in_channels, out_channels, p):
        super().__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.dropout = nn.Dropout(p=p)
        self.layer = TwoConv(in_channels, out_channels)

    def forward(self, x, x_prev):
        x = self.up(x)
        x = torch.cat([x_prev, x], dim=1)
        x = self.dropout(x)
        return self.layer(x)
    
class LastConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.layer(x)
    
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.first = TwoConv(n_channels, 64)
        self.contracting1 = Contracting(64, 128, dropout)
        self.contracting2 = Contracting(128, 256, dropout)
        self.contracting3 = Contracting(256, 512, dropout)
        self.contracting4 = Contracting(512, 1024, dropout)
        self.expansive1 = Expansive(1024, 512, dropout)
        self.expansive2 = Expansive(512, 256, dropout)
        self.expansive3 = Expansive(256, 128, dropout)
        self.expansive4 = Expansive(128, 64, dropout)
        self.last = LastConv(64, n_classes)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.contracting1(x1)
        x3 = self.contracting2(x2)
        x4 = self.contracting3(x3)
        x5 = self.contracting4(x4)
        x = self.expansive1(x5, x4)
        x = self.expansive2(x, x3)
        x = self.expansive3(x, x2)
        x = self.expansive4(x, x1)
        logits = self.last(x)
        return logits
