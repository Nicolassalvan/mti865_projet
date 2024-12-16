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



# Version du dice où image de prédiction est composée seulement de 1, là où la classe est le résultat 
# du argmax ou 0,sinon, à chaque pixel(bonne version niveau théorique je crois)
def dice_score_class3(model,dataloader):
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        dice_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        
        for i_batch in range (dice_target.shape[0]):
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                intersect3 = (pred_class*dice_target[i_batch][classe].data.numpy()).sum()+ smooth
                
                den3 = (((pred_class*pred_class).sum())+((dice_target[i_batch][classe].data.numpy()*dice_target[i_batch][classe].data.numpy()).sum()))+ smooth
                
                dice_class3=(2.0 * (intersect3 / den3))

                dice_class[classe]+= dice_class3
            nb_it+=1
            
    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class

# Meme version que dice_score_class3 mais cette fois c'est la soft dice (comme la soft dice loss) :
# ce n'est plus ŷp mais ŝp (softmax au lieu de binaire 0 ou 1)
def dice_score_class4(model,dataloader):
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        dice_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        
        for i_batch in range (dice_target.shape[0]):
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                pred_class = np.zeros(y_pred_batch.shape)
                for i in range (y_pred_batch.shape[0]):
                    for j in range (y_pred_batch.shape[1]):
                        if (y_pred_batch[i][j] == classe):
                            pred_class[i][j] = probs[i_batch][classe][i][j].item()

                intersect3 = (pred_class*dice_target[i_batch][classe].data.numpy()).sum()#+ smooth
                
                den3 = (((pred_class*pred_class).sum())+((dice_target[i_batch][classe].data.numpy()*dice_target[i_batch][classe].data.numpy()).sum()))#+ smooth
                
                dice_class3=(2.0 * (intersect3 / den3))

                if math.isnan(dice_class3): # utile seulement si on n'utilise pas de smooth dans intersect et den
                    dice_class3=0

                dice_class[classe]+= dice_class3
            nb_it+=1
            
    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class

# Même chose que dice_score_class3 (vraiment la même fonction) mais sans utiliser les smooth pour les
# intersections et les dénominateurs
def dice_score_class5(model,dataloader):
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        dice_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        
        for i_batch in range (dice_target.shape[0]):
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1


                intersect3 = (pred_class*dice_target[i_batch][classe].data.numpy()).sum()#+ smooth
                
                den3 = (((pred_class*pred_class).sum())+((dice_target[i_batch][classe].data.numpy()*dice_target[i_batch][classe].data.numpy()).sum()))#+ smooth
                
                dice_class3=(2.0 * (intersect3 / den3))

                if math.isnan(dice_class3): # utile seulement si on n'utilise pas de smooth dans intersect et den
                    dice_class3=0

                dice_class[classe]+= dice_class3
            nb_it+=1
            
    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class


# Meme version que dice_score_class5 mais utilise jaccard_score de sklearn.metrics, qui calcule 
# la métrique jaccard, et dice=2*jaccard*(dénominateur jaccard)/(dénominateur dice)
def dice_score_class6(model,dataloader):
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dice_class=[0,0,0,0]
    num_classes=4
    smooth=1
    nb_it=0
    nb_nan=[0,0,0,0]
    for idx, data in enumerate(dataloader):
        images, labels, img_names = data
        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        dice_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        
        for i_batch in range (dice_target.shape[0]):
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1


                intersect3 = (pred_class*dice_target[i_batch][classe].data.numpy()).sum()#+ smooth
                
                den3 = (((pred_class*pred_class).sum())+((dice_target[i_batch][classe].data.numpy()*dice_target[i_batch][classe].data.numpy()).sum()))#+ smooth
                den5=den3-intersect3

                jaccard=sklearn.metrics.jaccard_score(pred_class, dice_target[i_batch][classe].data.numpy(), average='micro')
                dice_class3=2*jaccard*den5/den3

                if math.isnan(dice_class3): # utile seulement si on n'utilise pas de smooth dans intersect et den
                    # dice_class3=0
                    nb_nan[classe]+=1
                    # dice_class3=0.01351351351
                    dice_class3=smooth

                dice_class[classe]+= dice_class3
            nb_it+=1

    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class


# Même version que dice_score_class3 mais en utilisant la fonction dc de medpy (qui calcule l'intersection 
# et le dénominateur et après le dice =2*intersection/den )
def dice_score_class7(model,dataloader): 
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        dice_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        for i_batch in range (dice_target.shape[0]):
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1
                dice_class[classe]+= dc(pred_class, dice_target[i_batch][classe].data.numpy())


            nb_it+=1

    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class


# Jaccard (or IOU) score with jaccard_score from sklearn.metrics (cette fois ci l'image prédiction est 
# composée seulement de 1, là où la classe est le résultat du argmax ou 0,sinon, à chaque pixel)
def jaccard_score_class4(model,dataloader):  
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        jaccard_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (jaccard_target.shape[0]):
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Jaccard
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                jaccard[classe]+= sklearn.metrics.jaccard_score(pred_class, jaccard_target[i_batch][classe].data.numpy(), average='micro')
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard


# Jaccard (or IOU) score avec ou sans smooth, sans fonction d'une librairie (image prédiction comme 
# jaccard_score_class4)
def jaccard_score_class5(model,dataloader): 
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        jaccard_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        for i_batch in range (jaccard_target.shape[0]):
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                # Calculate the intersection and denominator parts for the Jaccard
                intersect3 = (np.multiply(pred_class,jaccard_target[i_batch][classe].data.numpy())).sum()#+smooth
                
                den3 = ((np.multiply(pred_class,pred_class)) + (np.multiply(jaccard_target[i_batch][classe].data.numpy(), jaccard_target[i_batch][classe].data.numpy()))).sum()#+smooth
                den5=den3-intersect3#+smooth

                iou_c=(intersect3 / den5)
                if math.isnan(iou_c): # utile seulement si on n'utilise pas de smooth dans intersect et den
                    iou_c=0

                jaccard[classe]+= iou_c
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard


# Même fonction que jaccard_score_class5 avec iou_c=1 (au lieu de 0) si la valeur était aupravant nan 
def jaccard_score_class6(model,dataloader): 
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        jaccard_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        for i_batch in range (jaccard_target.shape[0]):
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                # Calculate the intersection and denominator parts for the Jaccard
                intersect3 = (np.multiply(pred_class,jaccard_target[i_batch][classe].data.numpy())).sum()#+smooth
                
                den3 = ((np.multiply(pred_class,pred_class)) + (np.multiply(jaccard_target[i_batch][classe].data.numpy(), jaccard_target[i_batch][classe].data.numpy()))).sum()#+smooth
                den5=den3-intersect3#+smooth

                iou_c=(intersect3 / den5)
                if math.isnan(iou_c): # utile seulement si on n'utilise pas de smooth dans intersect et den
                    iou_c=1

                jaccard[classe]+= iou_c
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard


# Même fonction que jaccard_score_class4 mais en utilisant jc from medpy au lieu de jaccard_score
# de sklearn.metrics
def jaccard_score_class7(model,dataloader):  
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        jaccard_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (jaccard_target.shape[0]):
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Jaccard
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                jaccard[classe]+= jc(pred_class, jaccard_target[i_batch][classe].data.numpy())
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard


# Même fonction que jaccard_score_class4 mais en utilisant binary_jaccard_index de torchmetrics au 
# lieu de jaccard_score de sklearn.metrics
def jaccard_score_class8(model,dataloader):  
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        jaccard_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (jaccard_target.shape[0]):
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Jaccard
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                jaccard[classe]+= binary_jaccard_index(tensor(pred_class), jaccard_target[i_batch][classe]).item()
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard



# Fonction pour la métrique HSD (disatnce Hausdorff) en utilisant hd from medpy 
def hsd_score_class(model,dataloader):  
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    hsd=[0,0,0,0]
    num_classes=4
    smooth=1e-6
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
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            hsd_target_argmax = torch.argmax(hsd_target, dim=1)
            _hd=0
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Jaccard
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