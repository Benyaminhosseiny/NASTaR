import numpy as np
import time
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# 1D Residual Block:
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# FullyConnected Block:
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

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Channel Attention:
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

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Efficient Channel Attention:
class eca_layer(nn.Module):
    # https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    """Constructs a ECA (Efficient Channel Attention) module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# CNN BackBones:
class CNNBackBone(nn.Module): # ResNet first 3 layers
    def __init__(self, model_name='resnet18', in_channels=3, pretrained=False, features_only=True):
        super( CNNBackBone, self ).__init__()
        CNNencoder        = timm.create_model(
            model_name    = model_name,
            in_chans      = in_channels,
            pretrained    = pretrained,
            features_only = features_only
            )
        self.CNNencoder = nn.Sequential( *list( CNNencoder.children() )[:7] ) # First 3 layers
        # self.CNNencoder = nn.Sequential( *list( CNNencoder.children() ) ) # First 3 layers
        #  NOTE:
        # For resnet: children 0,1,2,3 are initializations | children 4: layer-1 and so on ...

    def forward(self, x):
        x = self.CNNencoder(x)
        return x

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# ViT Architecture:
"""
  b: Batch size
  n: Number of tokens (patched sub-images)
  dim: Length of Embedding
  mlp_dim: Length of MLP neurons in Self-Attention (Transformer) module (Usually 4*dim)
  heads: attention heads
  depth: Transformer module length (Number of times that Transformer module repeats)

"""


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Ordinary or Cross-Attention MHSA:
class Attention(nn.Module): # Ordinary or Cross-Attention MHSA
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., cross_attention=True):
        super().__init__()
        inner_dim   = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.cross_attention = cross_attention
        self.heads  = heads
        self.scale  = dim_head ** -0.5                           # Scale-factor in the Eq: Attention(Q,K,V)=softmax(Q*K.T/(d_heads**0.5))*V

        self.norm   = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout= nn.Dropout(dropout)

        # If Ordinary MHSA:
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) # "*3" because of QKV
        
        # If Cross MHSA:
        self.to_kv  = nn.Linear(dim, inner_dim * 2, bias = False) # "*2" because of KV
        self.to_q   = nn.Linear(dim, inner_dim,     bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)                                                           # shape: (b, n+1, dim)
        
        if self.cross_attention == False:
            qkv   = self.to_qkv(x)                                                 # shape: (b, n+1, 3*dim)
            qkv   = qkv.chunk(3, dim = -1)                                         # A tuple with 3 items (q, k, v) each one: (b, n+1, dim)
            
            # Reshaping the QKV tokens based on the number of heads (dim -> headsxdim_heads): (b, n+1, dim) -> (b, heads, n+1, dim_heads)
            q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv) # shape: (b, heads, n+1, dim_heads)
        
        if self.cross_attention == True:
            q0      = self.to_q(x[:,0:1])                                            # shape: (b, 1, dim)
            kv   = self.to_kv(x)                                                   # shape: (b, n+1, 2*dim)
            kv   = kv.chunk(2, dim = -1)                                           # A tuple with 2 items (k, v) each one: (b, n+1, dim)

            # Reshaping the QKV tokens based on the number of heads (dim -> headsxdim_heads): (b, n+1, dim) -> (b, heads, n+1, dim_heads)
            # q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv) # shape: (b, heads, n+1, dim_heads)
            k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv) # shape: (b, heads, n+1, dim_heads)
            q   = rearrange(q0, 'b 1 (h d) -> b h 1 d', h=self.heads) # shape: (b, heads, 1, dim_heads)

        dots  = torch.matmul(q, k.transpose(-1, -2)) * self.scale                # shape: [ if Ordinary: (b, heads, n+1, n+1) if CA: (b, heads, 1, n+1) ] : Q*K.T/(d_heads**0.5)
        # same with the above line: dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn_w  = self.attend(dots)                                              # Attention weights: softmax(Q*K.T/(d_heads**0.5))
        attn_w  = self.dropout(attn_w)

        out   = torch.matmul(attn_w, v)                                          # shape: (b, heads, n+1, dim_heads) : Attention(Q,K,V)=softmax(Q*K.T/(d_heads**0.5))*V
        # Reshaping the tokens to their initial shape: (b, heads, n+1, dim_heads) -> (b, n+1, dim)
        out   = rearrange(out, 'b h n d -> b n (h d)')                           # shape: (b, n+1, heads*dim_heads)
        # same with the above lines: out = torch.einsum('bhij,bhjd->bhid', attn_w, v).transpose(1, 2); out = out.reshape(b, 1, dim_heads * self.heads)

        out   = self.to_out(out)                                                 # shape: (b, n+1, dim)

        return out, attn_w

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Transformer A.K.A Self-Attention Module
class Transformer(nn.Module): # A.K.A Self-Attention Module
    # Transformer block consists of: {(MHSA (qkv) -> MLP)}<-Repeat(depth) -> LayerNorm
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., cross_attention=True):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, cross_attention=cross_attention),
                FeedForward(dim, mlp_dim, dropout = dropout)                     # The default for mlp_dim is four times of embedding dim (4*dim)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_all=[]
        attn_w_all=[]
        for attn, ff in self.layers:
            attn_all  .append( attn(x)[0] )  # Attention module out: shape: (b, n+1, dim)
            attn_w_all.append( attn(x)[1] )  # Attention module weights: shape: (b, n+1, n+1)
            x = x + attn(x)[0]
            x = x + ff(x)
            x = self.norm(x) # B: Transferred the LayerNorm to all the layers!! (Use Shift+Tab to return it to its initial form!!)
        return x, attn_all, attn_w_all
    
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Ordinary ViT:
class ViT(nn.Module):
    """
      b:       Batch size
      n:       Number of tokens (patched sub-images)
      dim:     Length of Embedding
      mlp_dim: Length of MLP neurons in Self-Attention (Transformer) module (Usually 4*dim)
      heads:   Attention heads
      depth:   Transformer module length (Number of times that Transformer module repeats)

    """
    def __init__(self, *, 
                 image_size, 
                 patch_size, 
                 num_classes, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 pool = 'cls', 
                 channels = 3, 
                 dim_head = 64, 
                 dropout = 0., 
                 emb_dropout = 0.,
                 cross_attention=True
                 ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),                                           # vectorized patch from input image --> Embedding "dim"
            nn.LayerNorm(dim),
        )

        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))                # Classification token [will be concatenated to the Embeddings]
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # Position Embedding [Will be added to the Embeddings]
        self.dropout       = nn.Dropout(emb_dropout)

        self.transformer   = Transformer(dim,
                                         depth,
                                         heads,
                                         dim_head,
                                         mlp_dim,
                                         dropout,
                                         cross_attention)                                # (MHSA -> MLP)->Repeat(depth) : [Transformer module repeats "depth" times]

        self.pool          = pool
        self.to_latent     = nn.Identity()

        self.mlp_head      = nn.Linear(dim, num_classes)

    def forward(self, img):
        # 1- Input -> Patch -> vectorize -> Embed
        x = self.to_patch_embedding(img)                                         # shape: (b,n,dim) : Patch Embedding (Converting the input image into n smaller patches, then reshaping each patch into a 1d vector, then Embedding each vector into "dim"-sized vector)
        b, n, _ = x.shape

        # 2- Embeded feature vectors
        # -> Concatenate classification token
        # -> Add Position Embedding:
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # print(f"cls_tokens:{cls_tokens.shape}")
        x = torch.cat((cls_tokens, x), dim=1)                                    # shape: (b,n+1,dim) : Concatenating the cls token vector to the patched vectors => n+1
        x += self.pos_embedding[:, :(n + 1)]                                     # shape: (b,n+1,dim) : Including the position embedding to the embedded (feature values!!)
        x = self.dropout(x)                                                      # shape: (b,n+1,dim)

        # 3- Transformer module:
        x, attn, attn_w = self.transformer(x)                                    # shape: (b,n+1,dim) : Transformer's output.

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]                  # shape: (b,dim) : [Keep only the embedded vector corresponding to the cls token (1st vector)] Pooling ( for classification -cls-: keep only the feature by the cls token )

        x = self.to_latent(x)                                                    # shape: (b,dim) :Identity (latent)

        x = self.mlp_head(x)                                                     # shape: (b,N_classes) : Classifier
        out = {}
        out["classifier"]  = x                                                   # shape: (b,N_classes) : Classifier
        out["attention"]   = attn                                                # shape: (b,n+1,dim) : Attention (Transformer's output)
        out["attention_w"] = attn_w                                              # shape: (b,n+1,n+1) : Attention weights (Transformer's output)
        return out

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Physics-Informed ViT with Cross-Attention: 
# => PI encoded
# => with 1d residual
# => and channel attetion
# => and then add to vit as cls token
class ViT_PI_CA(nn.Module): # Physics-Informed ViT with Cross-Attention
    """
      b:       Batch size
      n:       Number of tokens (patched sub-images)
      dim:     Length of Embedding
      mlp_dim: Length of MLP neurons in Self-Attention (Transformer) module (Usually 4*dim)
      heads:   attention heads
      depth:   Transformer module length (Number of times that Transformer module repeats)
    """
    def __init__(self, *, 
                 image_size, 
                 patch_size, 
                 feat_channels, 
                 num_classes, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 pool = 'cls', 
                 channels = 3, 
                 dim_head = 64, 
                 dropout = 0., 
                 emb_dropout = 0.,
                 cross_attention=True
                 ):
        super().__init__()


        self.Res1d      = ResidualBlock1D(in_channels=feat_channels, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.CAttention = ChannelAttention(channel=dim)



        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim   = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),                                           # vectorized patch from input image --> Embedding "dim"
            nn.LayerNorm(dim),
        )

        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))                # Classification token [will be concatenated to the Embeddings]
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # Position Embedding [Will be added to the Embeddings]
        self.dropout       = nn.Dropout(emb_dropout)

        self.transformer   = Transformer(dim,
                                         depth,
                                         heads,
                                         dim_head,
                                         mlp_dim,
                                         dropout,
                                         cross_attention)                                # (MHSA -> MLP)->Repeat(depth) : [Transformer module repeats "depth" times]

        self.pool          = pool
        self.to_latent     = nn.Identity()

        self.mlp_head      = nn.Linear(dim, num_classes)

    def forward(self, img, feat):
        # 0. Features branch (Physical models)
        cls_tokens = self.Res1d(feat)
        cls_tokens = rearrange(cls_tokens, 'b d 1 -> b d 1 1')
        cls_tokens = self.CAttention(cls_tokens)

        # 1- Input -> Patch -> vectorize -> Embed
        x = self.to_patch_embedding(img)                                         # shape: (b,n,dim) : Patch Embedding (Converting the input image into n smaller patches, then reshaping each patch into a 1d vector, then Embedding each vector into "dim"-sized vector)
        b, n, _ = x.shape

        # 2- Embeded feature vectors
        # -> Concatenate classification token
        cls_tokens = rearrange(cls_tokens, 'b d 1 1 -> b 1 d')
        x = torch.cat((cls_tokens, x), dim=1)                                    # shape: (b,n+1,dim) : Concatenating the cls token vector to the patched vectors => n+1
        # -> Add Position Embedding:
        x += self.pos_embedding[:, :(n + 1)]                                     # shape: (b,n+1,dim) : Including the position embedding to the embedded (feature values!!)
        x = self.dropout(x)                                                      # shape: (b,n+1,dim)

        # 3- Transformer module:
        x, attn, attn_w = self.transformer(x)                                    # shape: (b,n+1,dim) : Transformer's output.

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]                  # shape: (b,dim) : [Keep only the embedded vector corresponding to the cls token (1st vector)] Pooling ( for classification -cls-: keep only the feature by the cls token )

        x = self.to_latent(x)                                                    # shape: (b,dim) :Identity (latent)

        x = self.mlp_head(x)                                                     # shape: (b,N_classes) : Classifier
        out = {}
        out["classifier"]  = x                                                   # shape: (b,N_classes) : Classifier
        out["attention"]   = attn                                                # shape: (b,n+1,dim) : Attention (Transformer's output)
        out["attention_w"] = attn_w                                              # shape: (b,n+1,n+1) : Attention weights (Transformer's output)
        return out

    
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def train_model(Model, optimizer, tr_loader, val_loader, train_losses, val_losses, scheduler, device, ep, PI_feat=False, es=None, W_losses={"CCE":1/3} ):
    """
    Train model.

    Args:
    Model       : Deep model.
    optimizer   : Optimizer for training.
    tr_loader   : Training data loader.
    val_loader  : Validation data loader.
    device      : Training device.
    ep          : Number of training epochs.
    PI_feat     : Model with Physical Features (PINN model)

    Returns:
    model        : Trained model.
    train_losses : Training losses.
    val_losses   : Validation losses.
    """
    val_loss_min     = np.inf
    no_loss_decrease = 0
    CCE = nn.CrossEntropyLoss() # (categorical pred (without Softmax), Labels (starting from 0, and not 1-hot encoded))

    def learning_1ep( Model, optimizer, device, dataloader, if_train = False, PI_feat=False, W_losses={"CCE":1/3} ):

        # Initialize loss values:
        cum_loss = {}
        cum_loss['loss'] = 0
        for k_ii in W_losses.keys():
          cum_loss[k_ii] = 0

        for X, X_feat, y_label in dataloader:
            X, X_feat, y_label = X.to(device), X_feat.to(device), y_label.to(device)
            if PI_feat:
                out_model  = Model(X, X_feat) # Out is a dictionary: "classifier", "attention"
            else:
                out_model  = Model(X) # Out is a dictionary: "classifier", "attention"

            
            pred_label = out_model["classifier"]

            # Calculate Losses:
            loss = 0
            Temp = 1 #0.2
            loss_label = CCE( pred_label/Temp, y_label )

            loss      += (W_losses["CCE"])*loss_label

            cum_loss['CCE']  += loss_label.item() # accumulated loss
            cum_loss['loss'] += loss.item() # accumulated loss

            torch.cuda.empty_cache()

            optimizer.zero_grad()

            if if_train:
                loss.backward()  # Backward pass (Compute Gradients)
                optimizer.step() # Update weights (parameters)

        return Model, optimizer, cum_loss

    # ==========================================================================================
    # ==========================================================================================

    N_tr  = len(tr_loader)
    N_val = len(val_loader)

    print( f"Number of mini-batches in training data: {N_tr} | Number of mini-batches in validation data: {N_val}" )

    # Training mode:
    Model.train()
    for ep_ii in range( 1, ep+1 ):
        pt = time.time()
        # initialize losses:
        train_loss = 0
        val_loss   = 0

        """ Training Phase """
        Model, optimizer, train_loss = learning_1ep( Model, optimizer, device, tr_loader, if_train=True, PI_feat=PI_feat,  W_losses=W_losses )

        """ Validation Phase """
        Model, optimizer, val_loss   = learning_1ep( Model, optimizer, device, val_loader, if_train=False, PI_feat=PI_feat, W_losses=W_losses )

        scheduler.step() # Update learning rate

        # Show the results:
        for key_ii in train_loss.keys():
            train_loss[key_ii] /= N_tr
            val_loss[key_ii]   /= N_val
            # Track losses:
            train_losses[key_ii].append(train_loss[key_ii])
            val_losses  [key_ii].append(val_loss  [key_ii])

        elapsed = time.time() - pt
        print( f"Epoch: {ep_ii} ... {elapsed: .5f} sec" )
        print( f"TRAINING MEAN LOSS   | Total: {train_loss['loss']: >8.5f} | CCE: {train_loss['CCE']: >8.5f}" )
        print( f"VALIDATION MEAN LOSS | Total: {val_loss['loss']: >8.5f} | CCE: {val_loss['CCE']: >8.5f}" )

        # Save model with lowest Validation loss:
        if val_loss['loss'] <= val_loss_min:
            print("============= Validation Loss DECREASED =============")
            best_weights = Model.state_dict()  # Save the current model weights
            # torch.save(Model.state_dict(), "Best_model")
            val_loss_min = val_loss['loss']
            no_loss_decrease = 0
        else:
            no_loss_decrease += 1

        if no_loss_decrease == es:
            print(f"============= training STOPPED at epoch {ep_ii} =============")
            break

    # Load the weights associated with the lowest loss
    Model.load_state_dict(best_weights)

    return Model, train_losses, val_losses

