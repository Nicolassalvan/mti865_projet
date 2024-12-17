from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from progressBar import printProgressBar
import scipy.io as sio
import pdb
import time
from os.path import isfile, join
import statistics
from PIL import Image
from medpy.metric.binary import dc, hd, asd, assd, jc
import scipy.spatial
from torch import tensor
from torchmetrics.functional.classification import binary_jaccard_index
import sklearn.metrics
import math
from utils import to_var,getTargetSegmentation
# from scipy.spatial.distance import directed_hausdorff


def Dice_score_class(model,dataloader,device,version):
    dice_class=[0,0,0,0]
    num_classes=4
    smooth=1e-6
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, img_names = data
        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        dice_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (dice_target.shape[0]):
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                # Calcule l'intersection et le denominateur pour le Dice score
                pred_class = np.zeros(y_pred_batch.shape)
                if(version!=3):
                    idx_pred = np.where(y_pred_batch == classe)
                    pred_class[idx_pred] = 1
                else :
                    for i in range (y_pred_batch.shape[0]):
                        for j in range (y_pred_batch.shape[1]):
                            if (y_pred_batch[i][j] == classe):
                                pred_class[i][j] = probs[i_batch][classe][i][j].item()
                
                if(version==1 or version ==2): # Calcul en dur du score dice
                    intersect = (pred_class*dice_target[i_batch][classe].data.numpy()).sum()#+ smooth
                    den = (((pred_class*pred_class).sum())+((dice_target[i_batch][classe].data.numpy()*dice_target[i_batch][classe].data.numpy()).sum()))#+ smooth
                    
                    if (version ==2):# Calcul en dur sans utiliser les smooth
                        #intersect=intersect-smooth
                        dice_classe=2.0 * (intersect / den)
                        if math.isnan(dice_classe): # utile si on n'utilise pas de smooth
                            dice_classe=0
                    else :
                        intersect=intersect+smooth
                        den=den+smooth
                        dice_classe=2.0 * (intersect / den)

                
                # Version qui prend plus longtemps à se calculer
                elif (version ==3): # Calcul en dur du score soft dice, ce n'est plus ŷp mais ŝp (softmax au lieu de binaire 0 ou 1)
                    intersect = (pred_class*dice_target[i_batch][classe].data.numpy()).sum()#+ smooth
                    den = (((pred_class*pred_class).sum())+((dice_target[i_batch][classe].data.numpy()*dice_target[i_batch][classe].data.numpy()).sum()))#+ smooth
                    
                    dice_classe=(2.0 * (intersect / den))
                    if math.isnan(dice_classe): # utile seulement si on n'utilise pas de smooth dans intersect et den
                        dice_classe=0

                elif (version ==4):# Calcul dice score avec medpy.binary.dc
                    dice_classe= dc(pred_class, dice_target[i_batch][classe].data.numpy())

                dice_class[classe]+= dice_classe
            nb_it+=1


    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class

def Jaccard_score_class(model,dataloader,device,version): 
    jaccard=[0,0,0,0]
    num_classes=4
    smooth=1e-6
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, img_names = data

        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        jaccard_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        for i_batch in range (jaccard_target.shape[0]):
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                # Calcule l'intersection et le denominateur pour le Jaccard score 
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1
                
                if (version==1):# Calcul en dur du jaccard score avec smooth
                    intersect = (np.multiply(pred_class,jaccard_target[i_batch][classe].data.numpy())).sum()+smooth
                
                    den = ((np.multiply(pred_class,pred_class)) + (np.multiply(jaccard_target[i_batch][classe].data.numpy(), jaccard_target[i_batch][classe].data.numpy()))).sum()+smooth
                    den5=den-intersect+smooth

                    jaccard_classe=(intersect / den5)

                elif(version==2):# Calcul en dur du jaccard score sans smooth
                    intersect = (np.multiply(pred_class,jaccard_target[i_batch][classe].data.numpy())).sum()
                
                    den = ((np.multiply(pred_class,pred_class)) + (np.multiply(jaccard_target[i_batch][classe].data.numpy(), jaccard_target[i_batch][classe].data.numpy()))).sum()
                    den5=den-intersect

                    jaccard_classe=(intersect / den5)
                    if math.isnan(jaccard_classe): # utile si on n'utilise pas de smooth dans intersect et den
                        jaccard_classe=0

                elif (version==3):# Calcul du jaccard score avec jc de medpy.binary
                    jaccard_classe= jc(pred_class, jaccard_target[i_batch][classe].data.numpy())

                elif (version==4):# Calcul du jaccard score avec binary_jaccard_index de torchmetrics
                    jaccard_classe= binary_jaccard_index(tensor(pred_class), jaccard_target[i_batch][classe]).item()
            
                jaccard[classe]+= jaccard_classe
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard


# Fonction pour la métrique HSD (disatnce Hausdorff) en utilisant hd from medpy 
def hsd_score_class(model,dataloader,device):  

    hsd=[0,0,0,0]
    num_classes=4
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, _ = data

        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        hsd_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (hsd_target.shape[0]):
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            _hd=0
            for classe in range (num_classes):
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                idx_target = np.where(hsd_target[i_batch][classe] == classe)
                # vérifie si la target pour chaque classe contient bien un objet (des 1 et pas que des 0)
                if len(idx_target[0])!=0 and len(idx_target[1])!=0 and len(idx_pred[0])!=0 and len(idx_pred[1])!=0 :
                    _hd=hd(pred_class, hsd_target[i_batch][classe].data.numpy())
                    hsd[classe]+= _hd

            nb_it+=1

    for classe in range (num_classes):
        hsd[classe]=hsd[classe]/nb_it

    # La distance de Hausdorff symétrique entre les objets prédits et les vrais objets. 
    # L'unité de distance est la même pour l'espacement des éléments dans chaque dimension, 
    # qui est généralement donné en mm.
    return hsd