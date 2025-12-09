
import rasterio as rio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings('ignore')

def tr_te_sample( gt_1d, tr_samples, val_samples, random_seed=1 ): 
  """
  Function to sample training, validation, and test indices from a 1D ground truth array
  gt_1d: 1D array of ground truth labels
  tr_samples: number of training samples per class (can be an integer or a fraction)
  val_samples: number of validation samples per class (can be an integer or a fraction)
  
  Returns: training labels, validation labels, test labels, training indices, validation indices, test indices
  """
  import numpy as np
  from copy import deepcopy
    
  np.random.seed(random_seed)
  tr_idx_L_ii   = []
  te_idx_L_ii   = []
  val_idx_L_ii  = []
    
  tr_label_L_ii  = []
  te_label_L_ii  = []
  val_label_L_ii = []
    
  for L_ii in range( 1, max(gt_1d)+1 ):
      
    L_ii_idx = np.where(gt_1d==L_ii)[0]
    np.random.shuffle(L_ii_idx)
      
    if tr_samples>1:
      tr_samples_ii  = deepcopy(tr_samples)
      val_samples_ii = deepcopy(val_samples)
    else:
      tr_samples_ii  = int( tr_samples*len(L_ii_idx) )
      val_samples_ii = int( val_samples*len(L_ii_idx) )


    tr_idx_L_ii .append(L_ii_idx[:tr_samples_ii])
    val_idx_L_ii.append(L_ii_idx[tr_samples_ii:tr_samples_ii+val_samples_ii])
    te_idx_L_ii .append(L_ii_idx[tr_samples_ii+val_samples_ii:])
    #
    tr_label_L_ii .append( L_ii*np.ones( len(L_ii_idx[:tr_samples_ii]) ) )
    val_label_L_ii.append( L_ii*np.ones( len(L_ii_idx[tr_samples_ii:tr_samples_ii+val_samples_ii]) ) )
    te_label_L_ii .append( L_ii*np.ones( len(L_ii_idx[tr_samples_ii+val_samples_ii:]) ) )
  #
  tr_label  = np.uint8( np.hstack(tr_label_L_ii) )
  val_label = np.uint8( np.hstack(val_label_L_ii) )
  te_label  = np.uint8( np.hstack(te_label_L_ii) )
  # 
  tr_idx  = np.uint32( np.hstack(tr_idx_L_ii) )
  val_idx = np.uint32( np.hstack(val_idx_L_ii) )
  te_idx  = np.uint32( np.hstack(te_idx_L_ii) )
  
  # Train:
  idx       = np.arange(0, len(tr_label), 1);  np.random.shuffle(idx)
  tr_label  = tr_label[idx]
  tr_idx    = tr_idx  [idx]
  # Validation:
  idx       = np.arange(0, len(val_label), 1); np.random.shuffle(idx)
  val_label = val_label[idx]
  val_idx   = val_idx  [idx]
  # Test:
  idx       = np.arange(0, len(te_label), 1); np.random.shuffle(idx)
  te_label  = te_label[idx]; te_idx=te_idx[idx]

  return tr_label, val_label, te_label, tr_idx, val_idx, te_idx

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def Rician_AnomalyDetection(image, AD_threshold=99):
    import numpy as np
    from scipy.stats import rice

    # 1. Preprocess image
    pixels = (image.flatten()).astype(np.float32)
    pixels /= np.max(pixels)  # Normalize to [0, 1]

    # 2. Estimate Rice distribution parameters using MLE
    # Rice distribution in scipy: rice(b, loc=0, scale=1)
    # b = nu / sigma, scale = sigma
    params = rice.fit(pixels, floc=0)  # Fix location to 0 for pixel intensities
    b_hat, loc_hat, sigma_hat = params
    nu_hat = b_hat * sigma_hat

    # 3. Compute likelihood for each pixel
    likelihoods = rice.pdf(pixels, b_hat, loc=loc_hat, scale=sigma_hat)

    # 4. Compute anomaly scores
    anomaly_scores = -np.log(likelihoods + 1e-12)  # Add epsilon to avoid log(0)

    # 5. Reshape to image shape
    anomaly_scores = anomaly_scores.reshape(image.shape)

    # 6. Threshold anomaly scores (e.g., top 1% most anomalous)
    threshold = np.percentile(anomaly_scores, AD_threshold)
    anomaly_mask = anomaly_scores > threshold
    anomaly_scores -= anomaly_scores.min()  # Normalize scores to start from 0
    anomaly_scores /= anomaly_scores.max()  # Normalize to [0, 1]
    return anomaly_scores, anomaly_mask, b_hat, loc_hat, sigma_hat

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ShipDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.num_classes = max(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        with rio.open(img_path) as src:
            img = src.read()  # shape: (channels, H, W)
            img = img.astype(np.float32)  # Convert to float32 for processing

            # Normalize the image [0-1]:
            img /= img.max()
            img -= img.mean()
            img /= 5*img.std()
            img += 0.5
            # img  = img.clip(0, 1) # Clip values to [0, 1] range

        tensor_img = torch.tensor(img, dtype=torch.float32)
        label_idx = self.labels[idx]-1

        # label_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        # label_onehot[label_idx ] = 1.0
        # return tensor_img, label_onehot
        return tensor_img, label_idx

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ShipDatasetMemory(Dataset):
    def __init__(self, images, features:None, labels: np.ndarray, transform=None):
        self.images = images
        self.features = features
        self.labels = labels
        self.num_classes = max(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        tensor_img  = torch.tensor(img, dtype=torch.float32)
        label_idx = self.labels[idx]-1

        if self.features is not None:
            feat = self.features[idx]
            tensor_feat = torch.tensor(feat, dtype=torch.float32)
        else:
            tensor_feat = torch.nan

        # Apply data augmentation if transform is provided
        if self.transform:
            
            if self.features is not None:
                ch_img  = 1 if tensor_img.ndim  == 2 else tensor_img.shape[0]
                ch_feat = 1 if tensor_feat.ndim == 2 else tensor_feat.shape[0]
                tensor_stack = torch.cat((tensor_img, tensor_feat), dim=0)  # Concatenate along the channel dimension
            
                tensor_stack = self.transform(tensor_stack)  # Apply transformation
                tensor_img  = tensor_stack[:ch_img, :, :]  # Extract the first channels as the image
                tensor_feat = tensor_stack[ch_img:, :, :]  # Extract the last channels as the feature
            else:
                tensor_img = self.transform(tensor_img)  # Apply transformation
        
        # label_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        # label_onehot[label_idx ] = 1.0
        # return tensor_img, label_onehot
        return tensor_img, tensor_feat,  label_idx

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Load images on memory

def load_images_to_memory(image_paths, dB_scale=False):
    images = []
    for img_pathii in tqdm(image_paths, desc="Loading images"):
        with rio.open(img_pathii) as src:
            imgii = src.read()  # shape: (channels, H, W)
            imgii = imgii.astype(np.float32)  # Ensure float32 type

            if dB_scale:
                imgii = np.log10(imgii + 1e-6)  # dB scale
            # Normalize the image [0-1]:
            imgii /= imgii.max()
            imgii -= imgii.mean()
            imgii /= 5*imgii.std()
            imgii += 0.5
            # imgii = imgii.clip(0, 1) # Clip values to [0, 1] range
            images.append(imgii)
    
    return np.array(images)

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def Encoded_images(encoder, im_dataloader, device):
    with torch.no_grad():
        X_encoded_all = []
        y_all = []
        for X_im, y in im_dataloader:
            X_im, y = X_im.to(device), y.to(device)
            X_encoded = encoder(X_im)
            X_encoded_all.append(X_encoded)#.cpu().numpy())
            y_all.append(y)#.cpu().numpy())
    return torch.cat(X_encoded_all), torch.cat(y_all)

class EncodedDatasetMemory(Dataset):
    def __init__(self, encoded_tnsr, labels, transform=None):
        self.encoded_tnsr = encoded_tnsr
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.encoded_tnsr)

    def __getitem__(self, idx):
        encoded_tnsrii = self.encoded_tnsr[idx]
        label_idx = self.labels[idx] # Note: In contrast to the ShipDatasetMemory, here we don't need to '-1' the labels because they are already in the range [1, N_classes]

        # Apply data augmentation if transform is provided
        if self.transform:
            encoded_tnsrii = self.transform(encoded_tnsrii)  

        return encoded_tnsrii, label_idx

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def Augmentation_Samples(images, features, labels, transformations, batch_size, rep=1):
    """ 
    images: numpy array (N, C, H, W)
    labels: numpy array (N,) 
    tranformations: done by pytorch transform class
    batch_size: batch size for DataLoader
    rep: number of repeating the process --> size of final samples is rep*N
    """
    _, label_counts = np.unique(labels, return_counts=True)
    
    # Apply data augmentation to training dataset
    dataset_augmented = ShipDatasetMemory( images=images, features=features, labels=labels, transform=transformations )
    
    # Create a new DataLoader for augmented training data
    dataloader_aug = DataLoader(dataset_augmented, batch_size=batch_size, shuffle=True)#, generator=g)
    
    # Generating samples:
    X_aug_all = []
    AD_score_aug_all = []
    y_aug_all = []
    for _ in range(rep):
        for X_aug, AD_score_aug, y_aug in dataloader_aug:
            X_aug_all.append( X_aug.numpy() )
            AD_score_aug_all.append( AD_score_aug.numpy() )
            y_aug_all.append( y_aug.numpy()+1 )
    X_aug_all = np.concatenate(X_aug_all)
    AD_score_aug_all = np.concatenate(AD_score_aug_all)
    y_aug_all = np.concatenate(y_aug_all)

    return X_aug_all, AD_score_aug_all, y_aug_all


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Custom Loss Functions for Imbalanced Data Training:
def class_weights_from_counts(class_counts, beta=0.999): 
    ''' Effective number of samples weighting (Cui et al., 2019) beta= 0.9~0.999
        class_counts: Class counts
    '''
    class_counts = torch.tensor(class_counts, dtype=torch.float32)
    eff_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / (eff_num + 1e-12)
    weights = weights / weights.mean()  # normalize for stability
    return weights

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def effective_weighted_ce(logits, targets, class_counts, beta=0.999):
    class_weights = class_weights_from_counts(class_counts, beta=beta).to(logits.device)
    return F.cross_entropy(logits, targets, weight=class_weights)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def logit_adjusted_ce(logits, targets, class_counts, tau=1.0):
    # class_counts: list/1D tensor of length C
    counts = torch.tensor(class_counts, dtype=torch.float32, device=logits.device)
    priors = counts / counts.sum()
    adjust = tau * torch.log(priors + 1e-12)  # tau ~ 1.0
    logits_adj = logits + adjust  # broadcast to batch
    return F.cross_entropy(logits_adj, targets)


class SoftMacroF1Loss(nn.Module):
    """
    Differentiable approximation of macro-F1 for multi-class classification.
    Uses softmax probabilities to compute soft TP/FP/FN per class over the batch.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C] pre-softmax
        targets: [B] int64 class indices in [0..C-1]
        returns: 1 - soft_macro_f1 (so it's a loss to minimize)
        """
        B, C = logits.shape
        probs    = F.softmax(logits, dim=1)               # [B, C]
        y_onehot = F.one_hot(targets.long(), num_classes=C).float()  # [B, C]

        # Soft counts over the batch
        tp = (probs * y_onehot).sum(dim=0)                      # [C]
        fp = (probs * (1.0 - y_onehot)).sum(dim=0)              # [C]
        fn = ((1.0 - probs) * y_onehot).sum(dim=0)              # [C]

        soft_f1_per_class = (2 * tp) / (2 * tp + fp + fn + self.eps)
        soft_macro_f1 = soft_f1_per_class.mean()

        return 1.0 - soft_macro_f1  # minimize
    
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def train_model(model, tr_dataloader, val_dataloader, N_classes, SM_temp, optimizer, scheduler_class, class_weights_tensor, num_epochs, num_reps, weight_export_name, physics_guided=False):
    # Training loop for model using tr_dataloader and val_dataloader
    device = class_weights_tensor.device
    # Loss Functions:
    BCELogitsLoss = torch.nn.BCEWithLogitsLoss(weight=class_weights_tensor)  # For multi-label one-hot targets
    # BCELogitsLoss = torch.nn.BCEWithLogitsLoss()#weight=class_weights_tensor)  # For multi-label one-hot targets
    CELoss = torch.nn.CrossEntropyLoss()#weight=class_weights_tensor)
    F1Loss = SoftMacroF1Loss()

    loss_history          = {}  # To store training and validation losses
    loss_history['train'] = []
    loss_history['val']   = []
    loss_condition_min = np.inf
    learning_rate0 = optimizer.param_groups[0]['lr']
    weight_decay0  = optimizer.param_groups[0].get('weight_decay', 0)
    for repii in range(num_reps):  # Repeat training for robustness
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        learning_rateii=learning_rate0/( repii+1 ) # Step decay of learning rate for each repetition
        # learning_rateii=learning_rate0*np.exp( -.5*repii ) # Exponential decay of learning rate for each repetition
        optimizerii = type(optimizer)(model.parameters(), lr=learning_rateii, weight_decay=weight_decay0)
        schedulerii = scheduler_class(optimizer)

        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for X_batch, AD_score_batch, y_batch in tr_dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if X_batch.shape[0] != 1:
                    optimizerii.zero_grad()

                    if physics_guided:
                        AD_score_batch = AD_score_batch.to(device)
                        pred_logits = model(X_batch, AD_score_batch)["classifier"]
                    else:
                        pred_logits = model(X_batch)["classifier"]
                    
                    if not isinstance(pred_logits, list): # If single output, convert to list (for ensemble consistency)
                        pred_logits = [pred_logits]
                    pred_logits = torch.stack(pred_logits, dim=0).mean(dim=0)/SM_temp  # Average ensemble outputs
                    
                    
                    ce_loss = BCELogitsLoss(pred_logits, F.one_hot(y_batch.long(), num_classes=N_classes).float())
                    # ce_loss = F.cross_entropy(pred_logits, y_batch, weight=class_weights_tensor)
                    # ce_loss = effective_weighted_ce(pred_logits, y_batch, class_counts=np.unique_counts(tr_label)[1], beta=0.99)
                    # ce_loss = logit_adjusted_ce(pred_logits, y_batch, class_counts=np.unique_counts(tr_label)[1])
                    # ce_loss = F.cross_entropy(pred_logits, y_batch)
                    f1_loss = F1Loss(pred_logits, y_batch)
                    
                    loss = ce_loss+0.005*f1_loss
                    
                    loss.backward()
                    optimizerii.step()
                    train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(tr_dataloader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, AD_score_val, y_val in val_dataloader:
                    X_val, y_val = X_val.to(device), y_val.to(device)

                    if physics_guided:
                        AD_score_val = AD_score_val.to(device)
                        pred_logits = model(X_val, AD_score_val)["classifier"]
                    else:
                        pred_logits = model(X_val)["classifier"]
                    if not isinstance(pred_logits, list): # If single output, convert to list (for ensemble consistency)
                        pred_logits = [pred_logits]
                    pred_logits = torch.stack(pred_logits, dim=0).mean(dim=0)/SM_temp  # Average ensemble outputs
                    
                    ce_loss = BCELogitsLoss(pred_logits, F.one_hot(y_val.long(), num_classes=N_classes).float())
                    # ce_loss = F.cross_entropy(pred_logits, y_val, weight=class_weights_tensor)
                    # ce_loss = effective_weighted_ce(pred_logits, y_val, class_counts=np.unique_counts(tr_label)[1], beta=0.99)
                    # ce_loss = logit_adjusted_ce(pred_logits, y_val, class_counts=np.unique_counts(tr_label)[1])
                    # ce_loss = F.cross_entropy(pred_logits, y_val)
                    f1_loss = F1Loss(pred_logits, y_val)
                    
                    loss = ce_loss+0.005*f1_loss
                    
                    val_loss += loss.item() * X_val.size(0)
            
            val_loss /= len(val_dataloader.dataset)
            
            schedulerii.step(val_loss)
            # exp_scheduler.step()
            
            loss_condition = (90*val_loss + 10*train_loss)/100  # Combined loss condition
            # loss_condition = 1*val_loss + 0*train_loss  # Combined loss condition
            
            if loss_condition <= loss_condition_min:

                print(f"Repeat {repii+1}: Epoch {epoch+1:3.0f}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} --> Loss condition decreased!")
                loss_condition_min = loss_condition
                # Save the model if validation loss improves
                best_weights = model.state_dict()
                if os.path.exists(weight_export_name):
                    os.remove(weight_export_name)
                torch.save(best_weights, weight_export_name)
            else:
                print(f"Repeat {repii+1}: Epoch {epoch+1:3.0f}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                if epoch % 1 == 0:
                    model.load_state_dict( torch.load(weight_export_name, map_location=device) )
            
            
            loss_history['train'].append(train_loss)
            loss_history['val']  .append(val_loss)
            
        model.load_state_dict( torch.load(weight_export_name, map_location=device) )

    return model, loss_history

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def model_inference(model, dataloader, label_names, device, physics_guided=False, show=False):
    from sklearn.metrics import confusion_matrix, accuracy_score
    import torch

    # Inference on training set
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, X_feat, y_batch in dataloader:
            X_batch = X_batch.to(device)
            if physics_guided:
                X_feat = X_feat.to(device)
                outputs = model(X_batch, X_feat)["classifier"]
            else:
                outputs = model(X_batch)["classifier"]
            if not isinstance(outputs, list): # If single output, convert to list (for ensemble consistency)
                        outputs = [outputs]
            outputs = torch.stack(outputs, dim=0).mean(dim=0)  # Average ensemble outputs

            preds  = torch.argmax(outputs, dim=1).cpu().numpy()
            # labels = torch.argmax(y_batch, dim=1).cpu().numpy()
            labels = y_batch.cpu().numpy()
            y_true.extend(labels)
            y_pred.extend(preds)

            if show:
                for Xii, Yii, Pii in zip(X_batch, labels, preds):
                    plt.figure(figsize=(3, 3))
                    plt.imshow(Xii.cpu().detach().numpy()[0], cmap='gray')
                    plt.title(f"Predicted Label: {Pii+1} {label_names[Pii]}\nTrue Label: {Yii+1} {label_names[Yii]}")
                    plt.axis('off')
                    plt.show()

    cm = confusion_matrix(y_true, y_pred)
    oa = accuracy_score(y_true, y_pred)
    
    return y_true, y_pred, cm, oa

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def confusion_mat(label, predicted, axlabels, plot=True, savefig_path="", cmap="gray_r"):
    #OA
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, recall_score, precision_score, f1_score
    import seaborn as sn
    import matplotlib.pyplot as plt
    import pandas as pd

    
    conf_mat    = np.float32(confusion_matrix(label, predicted))
    conf_mat_re = 100*conf_mat/np.expand_dims(np.nansum(conf_mat,axis=1),1) #Normalized Confusion Mat: Recall
    conf_mat_pr = 100*conf_mat/np.expand_dims(np.nansum(conf_mat,axis=0),0) #Normalized Confusion Mat: Precision
    conf_mat_f1 = 2*conf_mat_pr*conf_mat_re/(conf_mat_pr+conf_mat_re)    #Normalized Confusion Mat: f1-score
    # conf_mat_f1=(conf_mat/np.expand_dims(np.sum(conf_mat,axis=1),1)+conf_mat/np.expand_dims(np.sum(conf_mat,axis=0),1))/2
    conf_mat_f1[np.isnan(conf_mat_f1)==True]=0
    conf_mat_pr[np.isnan(conf_mat_pr)==True]=0
    print(f"conf_mat_pr.max: {np.nanmax(conf_mat_pr)}")
    
    OA       = 100*np.trace(conf_mat)/len(predicted)                    # OA
    Kappa    = 100*cohen_kappa_score(label, predicted)                  # Kappa
    bal_OA   = 100*recall_score(label, predicted, average='macro')      # Balanced Accuracy
    OPr      = 100*precision_score(label, predicted, average='macro')
    OF1      = 100*f1_score(label, predicted, average='macro')
    bal_OA_W = 100*recall_score(label, predicted, average='weighted')
    OPr_W    = 100*precision_score(label, predicted, average='weighted')
    OF1_W    = 100*f1_score(label, predicted, average='weighted')
    
    
    # print(f'OA          : {OA}')
    # print(f'Kappa       : {Kappa}')
    # print(f'Balanced OA : {bal_OA}')
    # print(f'Overall Pr  : {OPr}')
    # print(f'Overall F1  : {OF1}')
    
    #Heat Map:
    def plot_CM(CM, CM_thresh, xticklabels, yticklabels, num_decimals='.2f', cmap_vmin=None, cmap_vmax=None):
      sn.set(font_scale=1)#for label size
      # Annotation
      annots = []
      for ii in range(len(axlabels)):
        annots.append([])
        for jj in range(len(axlabels)):
          if CM[ii,jj]<CM_thresh:
            annots[-1] += ['']
          else:
            annots[-1] += [format( CM[ii,jj], num_decimals )]
      #
      ax=sn.heatmap(CM, annot=np.array(annots), annot_kws={"size": 10}, cbar=False, xticklabels=xticklabels, 
                    yticklabels=yticklabels, cmap=cmap, vmin=cmap_vmin, vmax=cmap_vmax, fmt = '',linewidths=.0)
      ax.axis('tight')
      # ax.axis('equal')
      
      # Frame
      lw=1
      ax.axhline(y=0, color='k',linewidth=lw); ax.axhline(y=len(axlabels), color='k',linewidth=lw)
      ax.axvline(x=0, color='k',linewidth=lw); ax.axvline(x=len(axlabels), color='k',linewidth=lw)
          
    
    #Other Evaluation Metrics:
    print('Data Evaluation:')
    print(classification_report(label, predicted))

    df = pd.DataFrame(data=[OA, Kappa, bal_OA, OPr, OF1],    # values
                              index=['OA', 'Kappa', 'BA', 'Pr', 'F1'],    # 1st column as index
                              columns=['Results'])  # 1st row as the column names
    display(df)

    if plot==True:
      plt.figure(figsize = (15, 15.45), facecolor='w', edgecolor='k')
      # plt.rcParams["font.family"] = "Times New Roman"
      # cmap=sn.cubehelix_palette(start=2, rot=0, dark=0, light=.9, reverse=True, as_cmap=True)
      # cmap="gray_r"
      # cmap="Greens"
      axlabels_num = []
      for numx in range(1,len(axlabels)+1):
        axlabels_num+=[str(numx)]
      #1
      plt.subplot(2,2,1);plt.title('Confusion matrix: OA='      +format( OA, '.4f' )    +' | K='+format( Kappa, '.4f' ),              fontsize=10)
      plot_CM(conf_mat, CM_thresh=0,  xticklabels=axlabels_num,  yticklabels=axlabels, num_decimals='.0f'  )
      #2
      plt.subplot(2,2,2);plt.title('F1: AF1='                   +format( OF1, '.4f' )   +' | AF1_Weighted='+format( OF1_W, '.4f' ),   fontsize=10)
      plot_CM(conf_mat_f1, CM_thresh=1, xticklabels=axlabels_num, yticklabels=axlabels_num, cmap_vmin=0, cmap_vmax=100 )
      #3
      plt.subplot(2,2,3);plt.title('Recall (Normalized CM): AA='+format( bal_OA, '.4f' )+' | AA_Weighted='+format( bal_OA_W, '.4f' ), fontsize=10)
      plot_CM(conf_mat_re, CM_thresh=1, xticklabels=axlabels_num, yticklabels=axlabels_num, cmap_vmin=0, cmap_vmax=100 )
      #4
      plt.subplot(2,2,4);plt.title('Precision: APr='            +format( OPr, '.4f' )   +' | APr_Weighted='+format( OPr_W, '.4f' ),   fontsize=10)
      plot_CM(conf_mat_pr, CM_thresh=1, xticklabels=axlabels_num, yticklabels=axlabels_num, cmap_vmin=0, cmap_vmax=100 )
      # plt.show()
      if savefig_path !="":
        plt.savefig(savefig_path)
    return conf_mat