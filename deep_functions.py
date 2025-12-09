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

class DeepModel(nn.Module):
    def __init__( self, in_channels, num_classes, BackBone, FC_input_dim, FC_neurons: List[int], num_ensemble=1, FC_dropout=[], input_CBAM=False ):
        super( DeepModel, self ).__init__()
        self.input_CBAM = input_CBAM
        if input_CBAM:
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

        else:
          neurons_en = [in_channels]+FC_neurons # if ResNet ignored!
        
        if FC_dropout==[]:
            FC_dropout=[0.0]*(len(neurons_en)-1)
        
        self.FC_branch = nn.ModuleList()
        for branchii in range( num_ensemble ):
            FC = []
            for ii in range( len(neurons_en)-1 ):
                FC.append( FCBlock( in_features=neurons_en[ii],
                                    out_features=neurons_en[ii+1],
                                    dropout=FC_dropout[ii]
                                )
                )
            
            FC.append( FCBlock( in_features=neurons_en[-1], out_features=num_classes, dropout=0.0  ) ) # Classifier
            FC = nn.Sequential( *FC )
            
            self.FC_branch.append( FC )

    def forward(self, x):
        if self.input_CBAM:
            CBAM_out = self.CBAM_attn(x) # (b,c,r,w)
        if self.with_cnn:
          if self.input_CBAM:
            x = self.cnn(CBAM_out)
            x = self.max_pool(x)
          else:
            x = self.cnn(x) # (b,f,r,w)
            x = self.max_pool(x) # (b,f,1,1)
        else:
          if self.input_CBAM:
            x = self.max_pool(CBAM_out)
          else:
            x = self.max_pool(x)
        
        x = torch.flatten(x,1)
        
        y = []
        for FCii in self.FC_branch:
            yii = FCii(x)
            y.append( yii )


        # Outputs:
        if self.input_CBAM:
            spatial_map = CBAM_out.max(dim=1, keepdim=True)[0]
            channel_map = CBAM_out.max(dim=2, keepdim=True)[0]
            channel_map = channel_map.max(dim=3, keepdim=True)[0]
        else:
            spatial_map = []
            channel_map = []

        out = {}
        out["classifier"]  = y # List of tensors (num_ensemble, (b,num_classes) )
        out["Encoder"]     = x
        out["spatial_map"] = spatial_map
        out["channel_map"] = channel_map
        return out
    

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# AlexNet:


class AlexNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(AlexNet, self).__init__()


        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
        

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        logits = self.classifier(x)
        
        out = {}
        out["classifier"]  = logits
        return out
    
    
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Physics-Guided Neural Networks (PGNNs):


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
# ================================================================================================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.relu(out)
    
# ================================================================================================================================

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, dilation):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
# ================================================================================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
# ================================================================================================================================

class MultiScaleFeatureExtractorWithAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFeatureExtractorWithAttention, self).__init__()
        self.dilate1 = DilatedConvBlock(in_channels, dilation=1)
        self.dilate2 = DilatedConvBlock(in_channels, dilation=2)
        self.dilate3 = DilatedConvBlock(in_channels, dilation=3)

        self.attention = ChannelAttention(in_channels)

    def forward(self, x):
        out1 = self.dilate1(x)
        out2 = self.dilate2(x)
        out3 = self.dilate3(x)

        concat = torch.cat([out1, out2, out3], dim=1)  # shape: (B, 3C, H, W)

        B, C_total, H, W = concat.shape
        C = C_total // 3
        pooled = torch.stack([
            concat[:, 0:C, :, :],
            concat[:, C:2*C, :, :],
            concat[:, 2*C:3*C, :, :]
        ], dim=0)

        max_pooled, _ = torch.max(pooled, dim=0)  # shape: (B, C, H, W)

        attended = self.attention(max_pooled)

        out = x + attended
        return out



class MultiScaleFeatureExtractorWithAttention(nn.Module):
    def __init__(self, in_channels, dilations):
        """
        Args:
            in_channels (int): Number of input channels.
            dilations (list of int): List of dilation rates for DilatedConvBlock.
        """
        super(MultiScaleFeatureExtractorWithAttention, self).__init__()
        
        self.Dil_Convs = nn.ModuleList([
            DilatedConvBlock(in_channels, dilation=dii)
            for dii in dilations
        ])
        
        self.attention = ChannelAttention(in_channels)

    def forward(self, x):
        # Apply each dilated convolution
        conv_outputs = [conv(x) for conv in self.Dil_Convs]

        # Concatenate along channel dimension
        concat = torch.cat(conv_outputs, dim=1)  # shape: (B, len(dilations)*C, H, W)

        # Split and stack for channel-wise max pooling
        C = x.shape[1]
        chunks = torch.stack([
            concat[:, i*C:(i+1)*C, :, :] for i in range(len(self.Dil_Convs))
        ], dim=0)  # shape: (num_dilations, B, C, H, W)

        max_pooled, _ = torch.max(chunks, dim=0)  # shape: (B, C, H, W)

        # Residual connection
        fused = x + max_pooled

        # Apply channel attention
        out = self.attention(fused)

        return out

# ================================================================================================================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels, modality_channels=1, heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.modality_proj = nn.Conv2d(modality_channels, in_channels, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x_main, x_modality):
        B, C, H, W = x_main.shape

        # Project modality to same channel dimension
        x_mod_proj = self.modality_proj(x_modality)  # shape: (B, C, H, W)

        # Flatten spatial dimensions for attention
        x_main_flat = x_main.view(B, C, -1).permute(0, 2, 1)     # (B, HW, C)
        x_mod_flat = x_mod_proj.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)

        # Apply cross-attention: query=x_modality, key=value=x_main
        fused, _ = self.attn(query=x_mod_flat, key=x_main_flat, value=x_main_flat)

        # Residual
        fused = fused + x_main_flat 

        # Layer normalization
        fused = self.norm(fused)

        # Reshape back to (B, C, H, W)
        fused = fused.permute(0, 2, 1).view(B, C, H, W)

        return fused

# ================================================================================================================================

# class StackedMultiScaleFeatureExtractor(nn.Module):
#     def __init__(self, in_channels, dilations, num_blocks):
#         """
#         Args:
#             in_channels (int): Number of input channels.
#             dilations (list of int): List of dilation rates for each multi-scale block.
#             num_blocks (int): Number of stacked blocks.
#         """
#         super(StackedMultiScaleFeatureExtractor, self).__init__()
        
#         self.blocks = nn.Sequential(*[
#             MultiScaleFeatureExtractorWithAttention(in_channels, dilations)
#             for _ in range(num_blocks)
#         ])

#     def forward(self, x):
#         return self.blocks(x)

# ================================================================================================================================

class StackedMultiModalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, dilations, num_blocks, modality_channels=1, heads=4):
        """
        Args:
            in_channels (int): Number of input channels for main modality.
            dilations (list of int): Dilation rates for multi-scale convolutions.
            num_blocks (int): Number of stacked blocks.
            modality_channels (int): Channels in the secondary modality (e.g., grayscale = 1).
            heads (int): Number of attention heads for cross-attention.
        """
        super(StackedMultiModalFeatureExtractor, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                MultiScaleFeatureExtractorWithAttention(in_channels, dilations),
                CrossAttentionFusion(in_channels, modality_channels, heads)
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x_main, x_modality):
        """
        Args:
            x_main (Tensor): Main input tensor of shape (B, C, H, W)
            x_modality (Tensor): Secondary modality tensor of shape (B, modality_channels, H, W)
        Returns:
            Tensor: Fused output of shape (B, C, H, W)
        """
        # for block in self.blocks:
        #     x_main = block0         # Multi-scale feature extraction
        #     x_main = block1  # Cross-attention fusion

        return self.blocks(x_main, x_modality)