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
from torchmetrics.functional.classification import binary_jaccard_index, binary_precision,binary_accuracy, binary_recall, binary_f1_score, binary_auroc
import sklearn.metrics
import math
from utils import to_var,getTargetSegmentation
# from scipy.spatial.distance import directed_hausdorff


# Fonction pour la métrique dice score en utilisant un calcul en dur des intersections et 
# dénominateur sans smooth. Renvoie un tableau contenant quatre dice score (un par classe)
def Dice_score_class(model,dataloader,device):
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
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1
                
                intersect = (pred_class*dice_target[i_batch][classe].data.numpy()).sum()#+ smooth
                den = (((pred_class*pred_class).sum())+((dice_target[i_batch][classe].data.numpy()*dice_target[i_batch][classe].data.numpy()).sum()))#+ smooth
                
                dice_classe=2.0 * (intersect / den)
                if math.isnan(dice_classe): # utile si on n'utilise pas de smooth
                    dice_classe=0


                dice_class[classe]+= dice_classe
            nb_it+=1


    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class

# Fonction pour la métrique jaccard score en utilisant binary_jaccard_index de torchmetrics. 
# Renvoie un tableau contenant quatre jaccard score (un par classe) 
def Jaccard_score_class(model,dataloader,device): 
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
                
                jaccard_classe= binary_jaccard_index(tensor(pred_class), jaccard_target[i_batch][classe]).item()
            
                jaccard[classe]+= jaccard_classe
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard


# Fonction pour la métrique HSD (disatnce Hausdorff) en utilisant hd de medpy. Renvoie un tableau
# contenant quatre HSD score (un par classe)
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


# Fonction pour la métrique precision en utilisant binary_precision de torchmetrics. Renvoie un 
# tableau contenant quatre precision score (un par classe) 
def precision_class(model,dataloader,device):  

    precision=[0,0,0,0]
    num_classes=4
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, _ = data

        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        precision_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (precision_target.shape[0]):
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                precision_classe=binary_precision(tensor(pred_class), precision_target[i_batch][classe]).item()
                precision[classe]+= precision_classe

            nb_it+=1

    for classe in range (num_classes):
        precision[classe]=precision[classe]/nb_it

    return precision


# Fonction pour la métrique accuracy en utilisant binary_accuracy de torchmetrics. Renvoie un 
# tableau contenant quatre accuracy score (un par classe) 
def accuracy_class(model,dataloader,device):  

    accuracy=[0,0,0,0]
    num_classes=4
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, _ = data

        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        accuracy_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (accuracy_target.shape[0]):
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                accuracy_classe=binary_accuracy(tensor(pred_class), accuracy_target[i_batch][classe]).item()
                accuracy[classe]+= accuracy_classe

            nb_it+=1

    for classe in range (num_classes):
        accuracy[classe]=accuracy[classe]/nb_it

    return accuracy

# Fonction pour la métrique recall en utilisant binary_recall de torchmetrics. Renvoie un tableau
# contenant quatre recall score (un par classe)
def recall_class(model,dataloader,device):  

    recall=[0,0,0,0]
    num_classes=4
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, _ = data

        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        recall_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (recall_target.shape[0]):
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                recall_classe=binary_recall(tensor(pred_class), recall_target[i_batch][classe]).item()
                recall[classe]+= recall_classe

            nb_it+=1

    for classe in range (num_classes):
        recall[classe]=recall[classe]/nb_it

    return recall


# Fonction pour la métrique f1 score en utilisant binary_f1_score de torchmetrics. Renvoie un 
# tableau contenant quatre f1 score (un par classe)
def f1_score_class(model,dataloader,device):  

    f1_score=[0,0,0,0]
    num_classes=4
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, _ = data

        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        f1_score_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (f1_score_target.shape[0]):
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                f1_score_classe=binary_f1_score(tensor(pred_class), f1_score_target[i_batch][classe]).item()
                f1_score[classe]+= f1_score_classe

            nb_it+=1

    for classe in range (num_classes):
        f1_score[classe]=f1_score[classe]/nb_it

    return f1_score

# Fonction pour la métrique AUC coeff en utilisant binary_auroc de torchmetrics. Renvoie un 
# tableau contenant quatre AUC coeff (un par classe) 
def auc_coeff_class(model,dataloader,device):  

    auc_coeff=[0,0,0,0]
    num_classes=4
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, _ = data

        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        auc_coeff_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (auc_coeff_target.shape[0]):
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                auc_coeff_classe=binary_auroc(tensor(pred_class), auc_coeff_target[i_batch][classe]).item()
                auc_coeff[classe]+= auc_coeff_classe

            nb_it+=1

    for classe in range (num_classes):
        auc_coeff[classe]=auc_coeff[classe]/nb_it

    return auc_coeff
