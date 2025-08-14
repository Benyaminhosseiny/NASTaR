import torch
from torch import nn
import timm
import torch.nn.functional as F

""" FC and CNN blocks """
class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, AF='relu', dropout=0.0):
        super( FCBlock, self ).__init__()
        self.fc      = nn.Linear(in_features, out_features) # Define a Fully-connected layer
        nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity=AF) # He initialization
        self.bn      = nn.BatchNorm1d(out_features)         # Add Batch-Normalization
        # self.af      = AF                                   # Add Non-linear function
        self.af      = getattr(F, AF)                       # Add Non-linear function
        self.dropout = nn.Dropout(p=dropout)                # Add Dropout

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.dropout(x)
        return x

# ================================================================================================================================

class CNNBackBone(nn.Module): # ResNet first 3 layers
    def __init__(self, model_name='resnet18', in_channels=3, pretrained=False, features_only=True):
        super( CNNBackBone, self ).__init__()
        CNNencoder        = timm.create_model(
            model_name    = model_name,
            in_chans      = in_channels,
            pretrained    = pretrained,
            features_only = features_only
            )
        # self.CNNencoder = nn.Sequential( *list( CNNencoder.children() )[:7] ) # First 3 layers
        self.CNNencoder = nn.Sequential( *list( CNNencoder.children() ) ) # First 3 layers
        #  NOTE:
        # For resnet: children 0,1,2,3 are initializations | children 4: layer-1 and so on ...


    def forward(self, x):
        x = self.CNNencoder(x)
        return x
    
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

""" Attention Mechanisms """

class ChannelAttention(nn.Module):
    # https://github.com/changzy00/pytorch-attention/blob/master/attention_mechanisms/cbam.py
    # Input shape: [b,c,r,w]
    def __init__(self, channel, reduction=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# ================================================================================================================================

class SpatialAttention(nn.Module):
    # https://github.com/changzy00/pytorch-attention/blob/master/attention_mechanisms/cbam.py
    # Input shape: [b,c,r,w]
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

# ================================================================================================================================

class CBAM(nn.Module):
    # Convolutional block attention (for a 3D spatial-spectral matrix)
    # https://github.com/changzy00/pytorch-attention/blob/master/attention_mechanisms/cbam.py
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
""" MODELS"""
from typing import Dict, List

class CNN(nn.Module):
    def __init__( self, in_channels, num_classes, BackBone, FC_input_dim, FC_neurons: List[int] ):
        super( CNN, self ).__init__()
        self.CBAM_attn = CBAM( channel=in_channels, reduction=1, kernel_size=3 )

        if BackBone !='':
          self.with_cnn = True
          self.cnn      = BackBone
        else:
          self.with_cnn = False

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Define Fully-connected:
        if self.with_cnn:
          neurons_en = [FC_input_dim]+FC_neurons # if with ResNet! and input patch dimensions: 512x512
          # neurons_en = [256]+FC_neurons
        else:
          neurons_en = [in_channels]+FC_neurons # if ResNet ignored!

        FC = []
        for ii in range( len(neurons_en)-1 ):
            FC.append( FCBlock( in_features=neurons_en[ii],
                                out_features=neurons_en[ii+1],
                                dropout=0.0
                              )
            )
        self.FC = nn.Sequential( *FC )


        # Define classifier:
        self.classifier = FCBlock( in_features=neurons_en[-1], out_features=num_classes, dropout=0.0  )

    def forward(self, x):
        CBAM_out = self.CBAM_attn(x)
        if self.with_cnn:
          x = self.cnn(CBAM_out)
          x = self.max_pool(x)
        else:
          x = self.max_pool(CBAM_out)
        
        x = torch.flatten(x,1)
        
        x = self.FC(x)

        # Classifier:
        y = self.classifier(x)

        # Outputs:
        spatial_map = CBAM_out.max(dim=1, keepdim=True)[0]
        channel_map = CBAM_out.max(dim=2, keepdim=True)[0]
        channel_map = channel_map.max(dim=3, keepdim=True)[0]

        out = {}
        out["classifier"]  = y
        out["Encoder"]     = x
        out["spatial_map"] = spatial_map
        out["channel_map"] = channel_map
        return out