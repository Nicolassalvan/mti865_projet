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
# from scipy.spatial.distance import directed_hausdorff


labels = {0: "Background", 1: "Foreground"}
LABEL_TO_COLOR = {
    0: [0, 0, 0], # Background
    1: [67, 67, 67], # Right ventricle (RV)
    2: [154, 154, 154], # Myocardium (MYO)
    3: [255, 255, 255] # Left ventricle (LV)
}



def compute_dsc(pred, gt):
    dsc_all = []
    # pdb.set_trace()
    for i_b in range(pred.shape[0]):
        pred_id = pred[i_b, 1, :]
        gt_id = gt[i_b, 0, :]
        dsc_all.append(dc(pred_id.cpu().data.numpy(), gt_id.cpu().data.numpy()))

    dsc = np.asarray(dsc_all)

    return dsc.mean()




def dice_score_class(model,dataloader):
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
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                intersect3 = torch.sum(torch.mul(probs[i_batch][classe], dice_target[i_batch][classe]), dim=(0,1))+ smooth
                
                den3 = torch.sum(probs[i_batch][classe].pow(2) + dice_target[i_batch][classe].pow(2), dim=(0,1))+ smooth
                
                dice_class3=(2.0 * (intersect3 / den3))

                dice_class[classe]+= dice_class3.item()
                # if nb_it==0 and classe==0:
                #     print("dice_class=",dice_class)
                
            nb_it+=1
            
    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class


def dice_score_class2(model,dataloader): # DICE score with dc from medpy
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
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                dice_class[classe]+= dc(probs[i_batch][classe].data.numpy(), dice_target[i_batch][classe].data.numpy())
                # if nb_it==0 and classe==0:
                #     print("dice_class2=",dice_class)
            nb_it+=1

    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class


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

                target_class = np.zeros(dice_target[i_batch][classe].shape)
                idx_target = np.where(dice_target[i_batch][classe] == 1)
                target_class[idx_target] = 1
                intersect3 = (pred_class*target_class).sum()+ smooth
                # intersect3 = torch.sum(torch.mul(probs[i_batch][classe], dice_target[i_batch][classe]), dim=(0,1))+ smooth
                
                den3 = (((pred_class*pred_class).sum())+((target_class*target_class).sum()))+ smooth
                # den3 = (((pred_class).sum())+((target_class).sum()))+ smooth
                
                dice_class3=(2.0 * (intersect3 / den3))
                # if nb_it==0 and classe==0:
                #     print("dice_class3=",dice_class3)

                dice_class[classe]+= dice_class3
            nb_it+=1
            
    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class

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
                # idx_pred = np.where(y_pred_batch == classe)
                # if nb_it==0 and classe==0:
                    # print("idx_pred=",idx_pred)
                    # print ("probs[i_batch][classe].size()=",probs[i_batch][classe].size())
                    # print("probs[i_batch][classe][idx_pred]=",probs[i_batch][classe][idx_pred])
                # pred_class[idx_pred] = 1

                # print ("y_pred_batch.shape[0]=",y_pred_batch.shape[0])
                # print ("y_pred_batch.shape[1]=",y_pred_batch.shape[1])
                # print ("probs[i_batch][classe][i][j]= ",probs[i_batch][classe][0][0])
                for i in range (y_pred_batch.shape[0]):
                    for j in range (y_pred_batch.shape[1]):
                        if (y_pred_batch[i][j] == classe):
                            d=0
                            pred_class[i][j] = probs[i_batch][classe][i][j].item()


                intersect3 = (pred_class*dice_target[i_batch][classe].data.numpy()).sum()#+ smooth
                # intersect3 = torch.sum(torch.mul(probs[i_batch][classe], dice_target[i_batch][classe]), dim=(0,1))+ smooth
                
                den3 = (((pred_class*pred_class).sum())+((dice_target[i_batch][classe].data.numpy()*dice_target[i_batch][classe].data.numpy()).sum()))#+ smooth
                # den3 = (((pred_class).sum())+((target_class).sum()))+ smooth
                
                dice_class3=(2.0 * (intersect3 / den3))
                # if nb_it==0 and classe==0:
                #     print("dice_class4=",dice_class3)

                if math.isnan(dice_class3): # utile seulement si on n'utilise pas de smooth dans intersect et den
                    dice_class3=0

                dice_class[classe]+= dice_class3
            nb_it+=1
            # print("nb_it=",nb_it)
            
    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class


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
                # intersect3 = torch.sum(torch.mul(probs[i_batch][classe], dice_target[i_batch][classe]), dim=(0,1))+ smooth
                
                den3 = (((pred_class*pred_class).sum())+((dice_target[i_batch][classe].data.numpy()*dice_target[i_batch][classe].data.numpy()).sum()))#+ smooth
                # den3 = (((pred_class).sum())+((target_class).sum()))+ smooth
                
                dice_class3=(2.0 * (intersect3 / den3))
                # if nb_it==0 and classe==0:
                #     print("dice_class3=",dice_class3)

                if math.isnan(dice_class3): # utile seulement si on n'utilise pas de smooth dans intersect et den
                    dice_class3=0

                dice_class[classe]+= dice_class3
            nb_it+=1
            
    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class


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
                # intersect3 = torch.sum(torch.mul(probs[i_batch][classe], dice_target[i_batch][classe]), dim=(0,1))+ smooth
                
                den3 = (((pred_class*pred_class).sum())+((dice_target[i_batch][classe].data.numpy()*dice_target[i_batch][classe].data.numpy()).sum()))#+ smooth
                # den3 = (((pred_class).sum())+((target_class).sum()))+ smooth
                den5=den3-intersect3

                jaccard=sklearn.metrics.jaccard_score(pred_class, dice_target[i_batch][classe].data.numpy(), average='micro')
                dice_class3=2*jaccard*den5/den3
                # dice_class3=(2.0 * (intersect3 / den3))
                # if nb_it==0 and classe==0:
                #     print("dice_class3=",dice_class3)

                if math.isnan(dice_class3): # utile seulement si on n'utilise pas de smooth dans intersect et den
                    dice_class3=0

                dice_class[classe]+= dice_class3
            nb_it+=1
            
    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class


def jaccard_score_class(model,dataloader): #Jaccard (or IOU) score
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    iou_class=[0,0,0,0]
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
        iou_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (iou_target.shape[0]):
            #Show first image and mask
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Jaccard (or IOU) coefficient
                intersect3 = torch.sum(torch.mul(probs[i_batch][classe], iou_target[i_batch][classe]), dim=(0,1))+ smooth
                
                den3 = torch.sum(probs[i_batch][classe].pow(2) + iou_target[i_batch][classe].pow(2), dim=(0,1))+ smooth
                den5=den3-intersect3

                iou_c=(intersect3 / den5)
                iou_class[classe]+= iou_c.item()
            nb_it+=1
            
    for classe in range (num_classes):
        iou_class[classe]=iou_class[classe]/nb_it
    return iou_class

def jaccard_score_class2(model,dataloader): #Jaccard (or IOU) score with binary jaccard coeff from torchmetrics
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
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                jaccard[classe]+= binary_jaccard_index(probs[i_batch][classe], jaccard_target[i_batch][classe],0.15).item()
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard


def jaccard_score_class3(model,dataloader):  #Jaccard (or IOU) score with jc from medpy
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
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                jaccard[classe]+= jc(probs[i_batch][classe].data.numpy(), jaccard_target[i_batch][classe].data.numpy())
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard


def jaccard_score_class4(model,dataloader):  #Jaccard (or IOU) score with jc from medpy
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
        jaccard_target_argmax = torch.argmax(jaccard_target, dim=1)

        for i_batch in range (jaccard_target.shape[0]):
            #Show first image and mask
            y_pred_batch = torch.argmax(probs[i_batch], dim=0)
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                pred_class = np.zeros(y_pred_batch.shape)
                idx_pred = np.where(y_pred_batch == classe)
                pred_class[idx_pred] = 1

                if nb_it==0 and classe==0 :
                    d=0
                    # print ("probs[i_batch][classe].data.numpy()=",probs[i_batch][classe].data.numpy())
                    # print ("y_pred[i_batch].data.numpy()=",y_pred[i_batch].data.numpy())
                    # print ("jaccard_target[i_batch][classe].data.numpy()=",jaccard_target[i_batch][classe].data.numpy())
                    # print ("jaccard_target_argmax[i_batch].data.numpy()=",jaccard_target_argmax[i_batch].data.numpy())
                    # print ("jaccard_target[i_batch][classe].data.numpy()=",jaccard_target[i_batch][classe].data.numpy())
                    # Enregistrer dans un fichier texte
                    np.savetxt("array_gt.txt", jaccard_target[i_batch][classe].data.numpy(), delimiter=",", fmt="%.2f", comments="")
                    np.savetxt("array_pred.txt", pred_class, delimiter=",", fmt="%.2f", comments="")

                    # Lecture pour vérifier
                    # with open("array_output.txt", "r") as file:
                    #     print(file.read())
                # jaccard[classe]+= sklearn.metrics.jaccard_score(y_pred[i_batch].data.numpy(), jaccard_target_argmax[i_batch].data.numpy(), average=None)
                jaccard[classe]+= sklearn.metrics.jaccard_score(pred_class, jaccard_target[i_batch][classe].data.numpy(), average='micro')
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard


def jaccard_score_class5(model,dataloader):  #Jaccard (or IOU) score with jc from medpy
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
    nb_nan=0
    nb_den5=0
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
                # Calculate the intersection and denominator parts for the Dice coefficient
                intersect3 = (np.multiply(pred_class,jaccard_target[i_batch][classe].data.numpy())).sum()#+smooth
                # intersect3 = ((pred_class* jaccard_target[i_batch][classe].data.numpy())).sum()+smooth
                
                den3 = ((np.multiply(pred_class,pred_class)) + (np.multiply(jaccard_target[i_batch][classe].data.numpy(), jaccard_target[i_batch][classe].data.numpy()))).sum()#+smooth
                # den3 = ((pred_class*pred_class) + (jaccard_target[i_batch][classe].data.numpy()* jaccard_target[i_batch][classe].data.numpy())).sum()+smooth
                den5=den3-intersect3#+smooth

                iou_c=(intersect3 / den5)
                if math.isnan(iou_c): # utile seulement si on n'utilise pas de smooth dans intersect et den
                    iou_c=0

                jaccard[classe]+= iou_c
                # jaccard[classe]+= jc(probs[i_batch][classe].data.numpy(), jaccard_target[i_batch][classe].data.numpy())
            nb_it+=1
    print ("nb_den5=",nb_den5)
    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard




def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
        imageNames = [
            f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))
        ]
        imageNames.sort()

    return imageNames


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    # pdb.set_trace()
    return (x == 1).float()


def plot_net_predictions(imgs, true_masks, masks_pred, batch_size):

    fig, ax = plt.subplots(3, batch_size, figsize=(20, 15))

    for i in range(batch_size):

        img = np.transpose(imgs[i].cpu().detach().numpy(), (1, 2, 0))
        mask_pred = masks_pred[i].cpu().detach().numpy()
        mask_true = np.transpose(true_masks[i].cpu().detach().numpy(), (1, 2, 0))

        ax[0, i].imshow(img, cmap="gray")
        ax[1, i].imshow(mask_to_rgb(mask_pred))
        ax[1, i].set_title("Predicted")
        ax[2, i].imshow(mask_true, cmap="gray")
        ax[2, i].set_title("Ground truth")

    return fig


def mask_to_rgb(mask):
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for i in np.unique(mask):
        rgb[mask == i] = LABEL_TO_COLOR[i]
    return rgb


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.33333334  # for ACDC this value
    return (batch / denom).round().long().squeeze()


def inference(net, img_batch, modelName, epoch):
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax().cuda()
    CE_loss = nn.CrossEntropyLoss().cuda()

    losses = []
    for i, data in enumerate(img_batch):

        printProgressBar(
            i, total, prefix="[Inference] Getting segmentations...", length=30
        )
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
        print(net_predictions.shape)
        segmentation_classes = getTargetSegmentation(labels)
        CE_loss_value = CE_loss(net_predictions, segmentation_classes)
        losses.append(CE_loss_value.cpu().data.numpy())
        pred_y = softMax(net_predictions)
        masks = torch.argmax(pred_y, dim=1)

        path = os.path.join("./Results/Images/", modelName, str(epoch))

        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(
            torch.cat(
                [
                    images.data,
                    labels.data,
                    masks.view(labels.shape[0], 1, 256, 256).data / 3.0,
                ]
            ),
            os.path.join(path, str(i) + ".png"),
            padding=0,
        )

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    losses = np.asarray(losses)

    return losses.mean()


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()
