
import rasterio as rio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import warnings
warnings.filterwarnings('ignore')

def tr_te_sample( gt_1d, tr_samples, val_samples ): 
  """
  Function to sample training, validation, and test indices from a 1D ground truth array
  gt_1d: 1D array of ground truth labels
  tr_samples: number of training samples per class (can be an integer or a fraction)
  val_samples: number of validation samples per class (can be an integer or a fraction)
  
  Returns: training labels, validation labels, test labels, training indices, validation indices, test indices
  """
  import numpy as np
  from copy import deepcopy
    
  np.random.seed(1)
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
        label_idx = self.labels[idx]

        label_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        label_onehot[label_idx - 1] = 1.0
        return tensor_img, label_onehot

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ShipDatasetMemory(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.num_classes = max(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label_idx = self.labels[idx]


        tensor_img = torch.tensor(img, dtype=torch.float32)

        # Apply data augmentation if transform is provided
        if self.transform:
            tensor_img = self.transform(tensor_img)
        
        label_onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        label_onehot[label_idx - 1] = 1.0
        return tensor_img, label_onehot
    
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Load images on memory

def load_images_to_memory(image_paths):
    images = []
    for img_pathii in tqdm(image_paths, desc="Loading images"):
        with rio.open(img_pathii) as src:
            imgii = src.read()  # shape: (channels, H, W)
            imgii = imgii.astype(np.float32)  # Ensure float32 type

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